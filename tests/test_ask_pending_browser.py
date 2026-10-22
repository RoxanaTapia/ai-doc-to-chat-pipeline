"""Browser checks for client ask-pending busy (Playwright).

Covers refusal follow-ups, empty Enter/click, right-click, failsafe restore,
and escaped HTML in answers. CI installs Chromium for this module.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEPLOY_DIR = PROJECT_ROOT / "deploy"
APP_PATH = SRC_DIR / "app.py"

TERM_QUESTION = "How long does this agreement last?"
FEE_QUESTION = "What is the penalty fee if Beta Labs breaks this agreement?"
LAW_QUESTION = "Which country's law applies, and where are disputes settled?"
READY_HINT = "Ask about this document"
LOOKING = "Looking through the document"

STUB_ENV = {
    "USE_STUB_EMBEDDINGS": "1",
    "DEMO_STUB_LLM": "1",
    "USE_DUMMY_GENERATOR": "1",
    "APP_PRESENTATION_MODE": "client",
    "TOKENIZERS_PARALLELISM": "false",
}


def _strip_boot_patch(text: str) -> str:
    """Remove a previous ai-doc boot injection so we can re-apply fresh assets."""
    import re

    text = re.sub(
        r"<!-- ai-doc-boot -->\s*<style>.*?</style>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<!-- ai-doc-boot -->\s*<div id=\"app-boot\".*?</script>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return text


@pytest.fixture(scope="session", autouse=True)
def _patch_streamlit_boot_into_index() -> Iterator[None]:
    """Local/CI Streamlit has no Docker boot patch; inject current CSS/JS for this suite."""
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))
    from patch_streamlit_index import _index_path, patch

    index = _index_path()
    original = _strip_boot_patch(index.read_text(encoding="utf-8"))
    index.write_text(original, encoding="utf-8")
    css = (DEPLOY_DIR / "streamlit_boot.css").read_text(encoding="utf-8")
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    patch(index, css, js)
    try:
        yield
    finally:
        index.write_text(original, encoding="utf-8")


def _import_playwright():
    try:
        from playwright.sync_api import expect
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        pytest.skip(f"Playwright not installed: {exc}")
    return sync_playwright, expect


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last_error = "no attempt"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
                last_error = f"status {response.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.3)
    raise RuntimeError(f"Streamlit not ready at {url}: {last_error}")


def _server_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(STUB_ENV)
    env.update(extra)
    env.pop("LLM_PROVIDER", None)
    env.pop("API_BASE_URL", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SRC_DIR)
    )
    return env


@contextmanager
def _streamlit_server(**extra_env: str) -> Iterator[str]:
    port = _free_port()
    env = _server_env(**extra_env)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}/app"
    try:
        _wait_http(url, timeout=90)
        yield url
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()


def _wait_ready(page, expect, timeout_ms: int = 60000) -> None:
    page.get_by_placeholder(READY_HINT).wait_for(state="visible", timeout=timeout_ms)
    expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled(timeout=timeout_ms)


def _pending(page) -> bool:
    return page.evaluate("() => document.body.classList.contains('app-ask-pending')")


def _ask_done_id(page) -> str:
    return page.evaluate(
        "() => (document.getElementById('app-ask-done') || {}).getAttribute?.('data-ask-id') || '0'"
    )


def _finished_cards(page) -> int:
    return page.locator('[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]').count()


def _ask_and_wait(page, expect, question: str, *, expect_text: str, min_cards: int) -> None:
    box = page.get_by_placeholder(READY_HINT)
    ask = page.get_by_role("button", name="Ask", exact=True)
    box.click()
    box.fill(question)
    expect(ask).to_be_enabled()
    before = _ask_done_id(page)
    ask.click()
    page.wait_for_function(
        """([before, minCards, expectText]) => {
          const done = document.getElementById('app-ask-done');
          const id = done ? done.getAttribute('data-ask-id') : '0';
          if (id === before) return false;
          if (document.body.classList.contains('app-ask-pending')) return false;
          const cards = document.querySelectorAll(
            '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
          );
          if (cards.length < minCards) return false;
          return (document.body.innerText || '').includes(expectText);
        }""",
        arg=[before, min_cards, expect_text],
        timeout=60000,
    )


def test_answer_then_refusal_clears_pending() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)
            assert not _pending(page)
            assert _finished_cards(page) >= 1

            _ask_and_wait(
                page,
                expect,
                FEE_QUESTION,
                expect_text="isn't in the document",
                min_cards=2,
            )
            assert not _pending(page)
            assert _finished_cards(page) >= 2
            assert page.evaluate(
                """() => {
                  const el = document.getElementById('app-ask-busy');
                  return !!el && getComputedStyle(el).display === 'none';
                }"""
            )
            browser.close()


def test_empty_enter_and_empty_click_do_not_mark_pending() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)
            cards_before = _finished_cards(page)

            box = page.get_by_placeholder(READY_HINT)
            box.click()
            box.fill("")
            box.press("Enter")
            page.wait_for_timeout(400)
            assert not _pending(page)
            assert _finished_cards(page) == cards_before

            page.get_by_role("button", name="Ask", exact=True).click()
            page.wait_for_timeout(400)
            assert not _pending(page)
            assert _finished_cards(page) == cards_before
            browser.close()


def test_right_click_ask_does_not_mark_pending() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)

            box = page.get_by_placeholder(READY_HINT)
            box.fill(LAW_QUESTION)
            ask = page.get_by_role("button", name="Ask", exact=True)
            ask.click(button="right")
            page.wait_for_timeout(400)
            assert not _pending(page)
            browser.close()


def test_failsafe_timer_restores_page() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.add_init_script("window.__APP_ASK_FAILSAFE_MS = 700;")
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)
            cards_before = _finished_cards(page)
            assert cards_before >= 1

            # Arm pending via the real helper; do not bump ask-done, so only the
            # failsafe timer can restore (unhide cards + drop client busy).
            page.evaluate("() => window.__appAskPending.markPending()")
            assert _pending(page)
            assert page.evaluate(
                "() => getComputedStyle(document.getElementById('app-ask-busy')).display !== 'none'"
            )
            page.wait_for_function(
                "() => !document.body.classList.contains('app-ask-pending')",
                timeout=5000,
            )
            assert not _pending(page)
            assert page.evaluate(
                "() => getComputedStyle(document.getElementById('app-ask-busy')).display === 'none'"
            )
            assert _finished_cards(page) >= cards_before
            browser.close()


def test_html_in_answer_is_escaped() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_HTML_ANSWER="1") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(
                page,
                expect,
                LAW_QUESTION,
                expect_text="payload about Spain",
                min_cards=1,
            )
            # Raw script/img nodes must not be injected from the answer body.
            assert page.locator(".app-answer-headline script, [class*='st-key-answer_card_'] script").count() == 0
            assert page.locator("[class*='st-key-answer_card_'] img[src='x']").count() == 0
            body = page.locator("body").inner_text()
            assert "<script>" in body or "alert(1)" in body
            assert "payload about Spain" in body
            browser.close()
