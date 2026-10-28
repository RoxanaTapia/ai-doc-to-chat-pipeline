"""Browser checks for client ask-pending busy (Playwright).

Runs against an unpatched Streamlit install so #207 does not depend on the
first-load index patch. Covers busy visibility in the status row,
refusal/answer/chip flows, empty Enter/click, right-click, failsafe via a
real submit, and escaped HTML.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
APP_PATH = DEMO_DIR / "app.py"

TERM_QUESTION = "How long does this agreement last?"
FEE_QUESTION = "What is the penalty fee if Beta Labs breaks this agreement?"
LAW_QUESTION = "Which country's law applies, and where are disputes settled?"
TERM_CHIP = "How long does this agreement last?"
READY_HINT = "Ask about this document"

STUB_ENV = {
    "USE_STUB_EMBEDDINGS": "1",
    "DEMO_STUB_LLM": "1",
    "USE_DUMMY_GENERATOR": "1",
    "APP_PRESENTATION_MODE": "client",
    "TOKENIZERS_PARALLELISM": "false",
}


def _ensure_streamlit_unpatched() -> None:
    """Strip any leftover first-load patch so #207 is proven on a stock install."""
    index = Path(st.__file__).resolve().parent / "static" / "index.html"
    text = index.read_text(encoding="utf-8")
    if "ai-doc-boot" in text:
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
        # Legacy external-script shape from an earlier PR revision.
        text = re.sub(
            r"<!-- ai-doc-boot -->\s*<script src=\"\./ai-doc-boot\.js\"></script>\s*",
            "",
            text,
            count=1,
        )
        index.write_text(text, encoding="utf-8")
        js_path = index.parent / "ai-doc-boot.js"
        if js_path.exists():
            js_path.unlink()
    text = index.read_text(encoding="utf-8")
    assert "ai-doc-boot" not in text
    assert "app-ask-pending" not in text
    assert not (index.parent / "ai-doc-boot.js").exists()


def _import_playwright():
    try:
        from playwright.sync_api import expect, sync_playwright
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def _busy_in_status_row(page) -> dict:
    """Whether the CSS busy line is visible inside the status row and on top."""
    return page.evaluate(
        """() => {
          const status = document.querySelector('.app-status');
          if (!status) return {ok: false, reason: 'no-status'};
          if (!document.body.classList.contains('app-ask-pending')) {
            return {ok: false, reason: 'not-pending'};
          }
          const r = status.getBoundingClientRect();
          if (r.width <= 0 || r.height <= 0) {
            return {ok: false, reason: 'zero-size', rect: [r.x, r.y, r.width, r.height]};
          }
          const before = getComputedStyle(status, '::before').content || '';
          if (!before.includes('Looking through the document')) {
            return {ok: false, reason: 'no-before', before};
          }
          const cx = r.left + Math.min(r.width / 2, 40);
          const cy = r.top + r.height / 2;
          const top = document.elementFromPoint(cx, cy);
          const onTop = !!top && (top === status || status.contains(top));
          if (!onTop) {
            return {
              ok: false,
              reason: 'covered',
              top: top && (top.tagName + (top.className ? '.' + top.className : '')),
              rect: [r.x, r.y, r.width, r.height],
            };
          }
          return {
            ok: true,
            rect: [r.x, r.y, r.width, r.height],
            before,
          };
        }"""
    )


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


def test_unpatched_streamlit_sets_busy_on_ask() -> None:
    """Fresh Streamlit install (no index patch): click still sets app-ask-pending."""
    _ensure_streamlit_unpatched()
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="80") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            # Client boot arrived via st.html, not /ai-doc-boot.js.
            assert page.evaluate("() => !!window.__appAskBootBound") is True
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)

            box = page.get_by_placeholder(READY_HINT)
            ask = page.get_by_role("button", name="Ask", exact=True)
            box.fill(LAW_QUESTION)
            expect(ask).to_be_enabled()
            ask.click()
            page.wait_for_function(
                "() => document.body.classList.contains('app-ask-pending')",
                timeout=100,
            )
            info = _busy_in_status_row(page)
            assert info.get("ok"), info
            page.wait_for_function(
                "() => !document.body.classList.contains('app-ask-pending')",
                timeout=60000,
            )
            browser.close()
    _ensure_streamlit_unpatched()


def test_busy_line_visible_in_status_row_within_100ms() -> None:
    _ensure_streamlit_unpatched()
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="80") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)

            box = page.get_by_placeholder(READY_HINT)
            ask = page.get_by_role("button", name="Ask", exact=True)
            box.fill(LAW_QUESTION)
            expect(ask).to_be_enabled()
            ask.click()
            page.wait_for_function(
                """() => {
                  if (!document.body.classList.contains('app-ask-pending')) return false;
                  const status = document.querySelector('.app-status');
                  if (!status) return false;
                  const r = status.getBoundingClientRect();
                  if (r.width <= 0 || r.height <= 0) return false;
                  const before = getComputedStyle(status, '::before').content || '';
                  if (!before.includes('Looking through the document')) return false;
                  const cx = r.left + Math.min(r.width / 2, 40);
                  const cy = r.top + r.height / 2;
                  const top = document.elementFromPoint(cx, cy);
                  return !!top && (top === status || status.contains(top));
                }""",
                timeout=100,
            )
            info = _busy_in_status_row(page)
            assert info.get("ok"), info
            assert info["rect"][2] > 0 and info["rect"][3] > 0
            # Stale rerun fade is opacity 1s ease-in with a 0.5s delay. Override it.
            fade = page.evaluate(
                """() => {
                  const el = document.querySelector('.stElementContainer:has(.app-status)');
                  if (!el) return null;
                  const cs = getComputedStyle(el);
                  return {
                    opacity: cs.opacity,
                    duration: cs.transitionDuration,
                    delay: cs.transitionDelay,
                  };
                }"""
            )
            assert fade == {"opacity": "1", "duration": "0s", "delay": "0s"}, fade
            # Status row sits below the hero, not at the page origin.
            assert info["rect"][1] > 40
            page.wait_for_function(
                "() => !document.body.classList.contains('app-ask-pending')",
                timeout=60000,
            )
            browser.close()


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
            browser.close()


def test_refusal_then_answer_clears_pending() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(
                page,
                expect,
                FEE_QUESTION,
                expect_text="isn't in the document",
                min_cards=1,
            )
            _ask_and_wait(page, expect, LAW_QUESTION, expect_text="Spain", min_cards=2)
            assert not _pending(page)
            assert "Spain" in page.locator("body").inner_text()
            browser.close()


def test_chip_click_marks_pending_and_answers() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="60") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            chip = page.get_by_role("button", name=TERM_CHIP, exact=True)
            expect(chip).to_be_visible(timeout=30000)
            chip.click()
            page.wait_for_function(
                """() => {
                  if (!document.body.classList.contains('app-ask-pending')) return false;
                  const status = document.querySelector('.app-status');
                  if (!status) return false;
                  const before = getComputedStyle(status, '::before').content || '';
                  return before.includes('Looking through the document');
                }""",
                timeout=100,
            )
            page.wait_for_function(
                """() => {
                  if (document.body.classList.contains('app-ask-pending')) return false;
                  return (document.body.innerText || '').includes('3 years');
                }""",
                timeout=60000,
            )
            assert not _pending(page)
            assert _finished_cards(page) >= 1
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


def test_failsafe_timer_restores_page_via_real_submit() -> None:
    """Failsafe arms on a real Ask click; fires before a slow answer can clear it."""
    sync_playwright, expect = _import_playwright()
    # Slow stub so ask-done cannot bump before the short failsafe fires.
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="250") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.add_init_script(
                "window.__APP_ASK_TEST_HOOKS = true; window.__APP_ASK_FAILSAFE_MS = 700;"
            )
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            _ask_and_wait(page, expect, TERM_QUESTION, expect_text="3 years", min_cards=1)
            cards_before = _finished_cards(page)

            box = page.get_by_placeholder(READY_HINT)
            box.fill(LAW_QUESTION)
            page.get_by_role("button", name="Ask", exact=True).click()
            page.wait_for_function(
                "() => document.body.classList.contains('app-ask-pending')",
                timeout=5000,
            )
            assert _busy_in_status_row(page).get("ok")
            page.wait_for_function(
                "() => !document.body.classList.contains('app-ask-pending')",
                timeout=5000,
            )
            assert not _pending(page)
            # Client busy CSS is gone; Streamlit may already have cleared the slot
            # for the in-flight ask, so do not require prior cards to still exist.
            assert not _busy_in_status_row(page).get("ok")
            before_css = page.evaluate(
                "() => getComputedStyle(document.querySelector('.app-status'), '::before').content"
            )
            assert "Looking through the document" not in (before_css or "")
            assert cards_before >= 1
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
            assert page.locator(
                ".app-answer-headline script, [class*='st-key-answer_card_'] script"
            ).count() == 0
            assert page.locator("[class*='st-key-answer_card_'] img[src='x']").count() == 0
            body = page.locator("body").inner_text()
            assert "<script>" in body or "alert(1)" in body
            assert "payload about Spain" in body
            browser.close()


_SIM_HTML = """<!DOCTYPE html>
<html><body>
  <div class="stElementContainer"><div class="app-status">Ready line</div></div>
  <div id="app-ask-done" data-ask-id="1" hidden></div>
  <div class="st-key-answer_card_1">Earlier answer</div>
  <div class="st-key-ask_submit"><button type="button">Ask</button></div>
  <div data-testid="stTextInput"><input type="text" /></div>
</body></html>
"""


def _raf_settle(page) -> None:
    page.evaluate(
        """() => new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        })"""
    )


def _open_pending_sim(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content(_SIM_HTML)
    page.evaluate("() => { window.__APP_ASK_TEST_HOOKS = true; }")
    page.add_script_tag(content=(DEMO_DIR / "ask_pending.js").read_text(encoding="utf-8"))
    page.wait_for_function("() => window.__appAskPending && window.__appAskBootBound")
    return browser, page


def test_done_marker_before_cards_does_not_clear() -> None:
    """Slot clear and a done id with no cards must keep the busy line up."""
    sync_playwright, _expect = _import_playwright()
    with sync_playwright() as playwright:
        browser, page = _open_pending_sim(playwright)
        page.evaluate("() => window.__appAskPending.markPending()")
        assert _pending(page)
        page.evaluate(
            """() => {
              const done = document.getElementById('app-ask-done');
              if (done) done.remove();
              const card = document.querySelector('.st-key-answer_card_1');
              if (card) card.remove();
            }"""
        )
        _raf_settle(page)
        assert _pending(page)

        page.evaluate(
            """() => {
              const done = document.createElement('div');
              done.id = 'app-ask-done';
              done.setAttribute('data-ask-id', '2');
              document.body.appendChild(done);
            }"""
        )
        _raf_settle(page)
        assert _pending(page)
        assert page.locator('[class*="st-key-answer_card_"]').count() == 0

        page.evaluate(
            """() => {
              const card = document.createElement('div');
              card.className = 'st-key-answer_card_2';
              card.textContent = 'New answer';
              document.body.appendChild(card);
            }"""
        )
        _raf_settle(page)
        assert _pending(page)

        page.evaluate(
            """() => {
              const card = document.createElement('div');
              card.className = 'st-key-answer_card_1';
              card.textContent = 'Earlier answer';
              document.body.appendChild(card);
            }"""
        )
        _raf_settle(page)
        assert not _pending(page)
        assert page.locator('[class*="st-key-answer_card_"]').count() == 2
        browser.close()


def test_cards_return_in_the_same_frame_as_the_answer() -> None:
    """Busy clears only once the new card and the earlier card are both present."""
    sync_playwright, _expect = _import_playwright()
    with sync_playwright() as playwright:
        browser, page = _open_pending_sim(playwright)
        page.evaluate("() => window.__appAskPending.markPending()")
        result = page.evaluate(
            """() => new Promise((resolve) => {
              let clearedEarly = false;
              const cardCount = () => document.querySelectorAll(
                '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
              ).length;
              const watch = new MutationObserver(() => {
                if (!document.body.classList.contains('app-ask-pending') && cardCount() < 2) {
                  clearedEarly = true;
                }
              });
              watch.observe(document.body, {attributes: true, childList: true, subtree: true});
              const oldDone = document.getElementById('app-ask-done');
              if (oldDone) oldDone.remove();
              const oldCard = document.querySelector('.st-key-answer_card_1');
              if (oldCard) oldCard.remove();
              const newer = document.createElement('div');
              newer.className = 'st-key-answer_card_2';
              newer.textContent = 'New answer';
              const earlier = document.createElement('div');
              earlier.className = 'st-key-answer_card_1';
              earlier.textContent = 'Earlier answer';
              const done = document.createElement('div');
              done.id = 'app-ask-done';
              done.setAttribute('data-ask-id', '2');
              document.body.appendChild(newer);
              document.body.appendChild(earlier);
              document.body.appendChild(done);
              requestAnimationFrame(() => requestAnimationFrame(() => {
                resolve({
                  pending: document.body.classList.contains('app-ask-pending'),
                  cards: cardCount(),
                  clearedEarly,
                });
              }));
            })"""
        )
        assert result["clearedEarly"] is False
        assert result["pending"] is False
        assert result["cards"] == 2
        browser.close()


def test_enter_sets_pending_for_typed_value_without_ask_button() -> None:
    """Enter marks pending even if the Ask button is disabled or the value lands on keyup."""
    sync_playwright, _expect = _import_playwright()
    with sync_playwright() as playwright:
        browser, page = _open_pending_sim(playwright)
        page.evaluate(
            """() => {
              const button = document.querySelector('[class*="st-key-ask_submit"] button');
              button.disabled = true;
              button.setAttribute('aria-disabled', 'true');
              const input = document.querySelector('[data-testid="stTextInput"] input');
              input.value = 'What is the penalty fee if Beta Labs breaks this agreement?';
              input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
            }"""
        )
        assert _pending(page)
        page.evaluate("() => window.__appAskPending.clearPending()")
        assert not _pending(page)

        page.evaluate(
            """() => {
              const input = document.querySelector('[data-testid="stTextInput"] input');
              input.value = '';
              input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
              input.value = 'What is the penalty fee if Beta Labs breaks this agreement?';
              input.dispatchEvent(new KeyboardEvent('keyup', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
            }"""
        )
        assert _pending(page)
        page.evaluate("() => window.__appAskPending.clearPending()")
        page.evaluate(
            """() => {
              const input = document.querySelector('[data-testid="stTextInput"] input');
              input.value = '   ';
              input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
              input.dispatchEvent(new KeyboardEvent('keyup', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
            }"""
        )
        assert not _pending(page)
        browser.close()


def test_typed_enter_on_first_refusal_shows_busy_line() -> None:
    """A typed Enter on a refusal, as the first question, sets the busy line."""
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="120") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            box = page.get_by_placeholder(READY_HINT)
            box.click()
            box.type(FEE_QUESTION, delay=0)
            box.press("Enter")
            page.wait_for_function(
                "() => document.body.classList.contains('app-ask-pending')",
                timeout=100,
            )
            info = _busy_in_status_row(page)
            assert info.get("ok"), info
            page.wait_for_function(
                """() => {
                  if (document.body.classList.contains('app-ask-pending')) return false;
                  const text = document.body.innerText || '';
                  return text.includes("isn't in the document");
                }""",
                timeout=60000,
            )
            assert not _pending(page)
            assert _finished_cards(page) >= 1
            browser.close()


def test_cap_keeps_busy_until_earlier_cards_return() -> None:
    """At the card cap, dropping the oldest card still clears once the rest are back."""
    sync_playwright, _expect = _import_playwright()
    with sync_playwright() as playwright:
        browser, page = _open_pending_sim(playwright)
        page.evaluate(
            """() => {
              const done = document.getElementById('app-ask-done');
              done.setAttribute('data-ask-id', '20');
              done.setAttribute('data-ask-cap', '20');
              const seeded = document.querySelector('.st-key-answer_card_1');
              if (seeded) seeded.remove();
              for (let n = 20; n > 0; n -= 1) {
                const card = document.createElement('div');
                card.className = 'st-key-answer_card_' + n;
                card.textContent = 'card ' + n;
                document.body.appendChild(card);
              }
              window.__appAskPending.markPending();
            }"""
        )
        assert _pending(page)
        page.evaluate(
            """() => {
              document.querySelectorAll(
                '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
              ).forEach((node) => node.remove());
              const oldDone = document.getElementById('app-ask-done');
              if (oldDone) oldDone.remove();
              const done = document.createElement('div');
              done.id = 'app-ask-done';
              done.setAttribute('data-ask-id', '21');
              done.setAttribute('data-ask-cap', '20');
              document.body.appendChild(done);
              const fresh = document.createElement('div');
              fresh.className = 'st-key-answer_card_21';
              fresh.textContent = 'card 21';
              document.body.appendChild(fresh);
            }"""
        )
        _raf_settle(page)
        assert _pending(page)

        page.evaluate(
            """() => {
              for (let n = 20; n > 1; n -= 1) {
                const card = document.createElement('div');
                card.className = 'st-key-answer_card_' + n;
                card.textContent = 'card ' + n;
                document.body.appendChild(card);
              }
            }"""
        )
        _raf_settle(page)
        assert not _pending(page)
        assert page.locator(".st-key-answer_card_1").count() == 0
        assert page.locator(".st-key-answer_card_21").count() == 1
        assert page.locator('[class*="st-key-answer_card_"]').count() == 20
        browser.close()


def test_repeat_enter_does_not_reset_failsafe() -> None:
    """A second Enter while pending must not push the 30s failsafe back."""
    sync_playwright, _expect = _import_playwright()
    with sync_playwright() as playwright:
        browser, page = _open_pending_sim(playwright)
        page.evaluate(
            """() => {
              window.__APP_ASK_FAILSAFE_MS = 1000;
              const input = document.querySelector('[data-testid="stTextInput"] input');
              input.value = 'How long does this agreement last?';
              input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
            }"""
        )
        assert _pending(page)
        page.wait_for_timeout(700)
        page.evaluate(
            """() => {
              const input = document.querySelector('[data-testid="stTextInput"] input');
              input.dispatchEvent(new KeyboardEvent('keyup', {
                key: 'Enter', bubbles: true, cancelable: true
              }));
            }"""
        )
        assert _pending(page)
        page.wait_for_timeout(500)
        assert not _pending(page)
        browser.close()


def _cap_question(n: int) -> str:
    token = f"slot-{n:02d}"
    if n % 8 == 0:
        return f"What is the penalty fee for {token}?"
    return f"How long does {token} last?"


def test_twenty_four_asks_clear_busy_line_at_history_cap() -> None:
    """Past 20 cards the oldest turn is dropped and the busy line still clears."""
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            box = page.get_by_placeholder(READY_HINT)
            ask = page.get_by_role("button", name="Ask", exact=True)
            for n in range(1, 25):
                question = _cap_question(n)
                before = _ask_done_id(page)
                page.evaluate(
                    """(question) => {
                      window.__askGap = false;
                      window.__askSawPending = false;
                      const hasQuestion = () => Array.from(document.querySelectorAll(
                        '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
                      )).some((card) => (card.textContent || '').includes(question));
                      const note = () => {
                        const pending = document.body.classList.contains('app-ask-pending');
                        if (pending) window.__askSawPending = true;
                        if (window.__askSawPending && !pending && !hasQuestion()) {
                          window.__askGap = true;
                        }
                      };
                      if (window.__askGapObs) window.__askGapObs.disconnect();
                      window.__askGapObs = new MutationObserver(note);
                      window.__askGapObs.observe(document.body, {
                        attributes: true,
                        childList: true,
                        subtree: true,
                        characterData: true,
                      });
                    }""",
                    question,
                )
                box.click()
                box.fill(question)
                expect(ask).to_be_enabled()
                ask.click()
                page.wait_for_function(
                    """(question) => {
                      if (window.__askGap) return false;
                      if (document.body.classList.contains('app-ask-pending')) return false;
                      return Array.from(document.querySelectorAll(
                        '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
                      )).some((card) => (card.textContent || '').includes(question));
                    }""",
                    arg=question,
                    timeout=3000,
                )
                assert page.evaluate("() => window.__askSawPending === true")
                assert page.evaluate("() => window.__askGap === false")
                assert not _pending(page)
                assert _ask_done_id(page) != before
                assert _finished_cards(page) == min(n, 20)
            body = page.locator("body").inner_text()
            assert "slot-01" not in body
            assert "slot-05" in body
            assert "slot-24" in body
            browser.close()


def test_type_ask_once_then_enter_keeps_every_card() -> None:
    """Type and click Ask once, then Enter. Each question shows one card, and earlier cards stay.

    The stub stream pauses so the Ask click can land while the model call is still
    running, which is the rerun that used to drop the look-up. Starter chips are
    on screen until the first answer; after that they hide, so the chip is clicked
    first and its card has to survive the typed asks.
    """
    _ensure_streamlit_unpatched()
    sync_playwright, expect = _import_playwright()
    with _streamlit_server(DEMO_STUB_STREAM_DELAY_MS="25") as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.goto(url, wait_until="domcontentloaded")
            _wait_ready(page, expect)
            expect(page.get_by_role("button", name=TERM_QUESTION, exact=True)).to_be_visible()

            page.get_by_role("button", name=TERM_QUESTION, exact=True).click()
            page.wait_for_function(
                """(expectText) => {
                  if (document.body.classList.contains('app-ask-pending')) return false;
                  const cards = document.querySelectorAll(
                    '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
                  );
                  return cards.length >= 1 && (document.body.innerText || '').includes(expectText);
                }""",
                arg="3 years",
                timeout=60000,
            )
            assert _finished_cards(page) == 1

            _ask_and_wait(page, expect, LAW_QUESTION, expect_text="Spain", min_cards=2)
            assert _finished_cards(page) == 2
            assert "3 years" in page.locator("body").inner_text()

            box = page.get_by_placeholder(READY_HINT)
            before = _ask_done_id(page)
            box.click()
            box.fill(FEE_QUESTION)
            box.press("Enter")
            page.wait_for_function(
                """([before, expectText]) => {
                  const done = document.getElementById('app-ask-done');
                  const id = done ? done.getAttribute('data-ask-id') : '0';
                  if (id === before) return false;
                  if (document.body.classList.contains('app-ask-pending')) return false;
                  const cards = document.querySelectorAll(
                    '[class*="st-key-answer_card_"], [class*="st-key-refusal_card_"]'
                  );
                  return cards.length >= 3 && (document.body.innerText || '').includes(expectText);
                }""",
                arg=[before, "isn't in the document"],
                timeout=60000,
            )
            body = page.locator("body").inner_text()
            assert _finished_cards(page) == 3
            assert "3 years" in body
            assert "Spain" in body
            assert "isn't in the document" in body
            assert not _pending(page)
            browser.close()
    _ensure_streamlit_unpatched()
