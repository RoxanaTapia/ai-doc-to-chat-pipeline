"""Browser checks for first-load loader / chip gate (production boot patch).

Applies the same ``deploy/patch_streamlit_index.py`` step as the Dockerfile, then
asserts the #app-boot loader covers the first paint, skeletons stay hidden, and
starter chips at 390 px do not jump.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
DEPLOY_DIR = PROJECT_ROOT / "deploy"
APP_PATH = DEMO_DIR / "app.py"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from page_copy import _BOOKING_PAGE_URL  # noqa: E402

READY_HINT = "Ask about this document"
TERM_CHIP = "How long does this last?"

STUB_ENV = {
    "USE_STUB_EMBEDDINGS": "1",
    "DEMO_STUB_LLM": "1",
    "USE_DUMMY_GENERATOR": "1",
    "APP_PRESENTATION_MODE": "client",
    "TOKENIZERS_PARALLELISM": "false",
}


def _strip_boot_patch(text: str) -> str:
    """Remove a previous ai-doc boot injection so the fixture can patch cleanly."""
    text = re.sub(
        r"<!-- ai-doc-boot -->.*?<!-- /ai-doc-boot -->\s*",
        "",
        text,
        flags=re.DOTALL,
    )
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


@pytest.fixture(scope="module")
def _production_boot_patch() -> Iterator[None]:
    """Match the Dockerfile: patch Streamlit's index with boot CSS/JS, then restore."""
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))
    from patch_streamlit_index import _index_path, install_server_hook, patch

    index = _index_path()
    original = _strip_boot_patch(index.read_text(encoding="utf-8"))
    index.write_text(original, encoding="utf-8")
    css = (DEPLOY_DIR / "streamlit_boot.css").read_text(encoding="utf-8")
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    assert patch(index, css, js) is True
    hooked = index.parent  # kit is copied beside index.html
    assert (hooked / "demo-kit" / "theme-init.js").is_file()
    import streamlit

    routes = Path(streamlit.__file__).resolve().parent / "web" / "server" / "routes.py"
    ops_copy = routes.parent / "ai_doc_ops.py"
    original_routes = routes.read_text(encoding="utf-8")
    had_ops = ops_copy.exists()
    install_server_hook()
    try:
        yield
    finally:
        routes.write_text(original_routes, encoding="utf-8")
        if not had_ops and ops_copy.exists():
            ops_copy.unlink()
        index.write_text(original, encoding="utf-8")


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


def _sample_first_paint(page, *, max_ms: int = 8000) -> dict:
    """Sample frames from navigation until the ask shell is ready."""
    return page.evaluate(
        """(maxMs) => new Promise((resolve) => {
          const started = performance.now();
          const samples = [];
          let sawBoot = false;
          let sawVisibleSkeleton = false;
          let sawUnstyledDeploy = false;
          let blankFrames = 0;
          let whiteFrames = 0;
          let bootBg = "";

          const tick = () => {
            const boot = document.getElementById('app-boot');
            const bootVisible = !!(
              boot
              && !boot.hidden
              && boot.dataset.done !== '1'
              && getComputedStyle(boot).display !== 'none'
              && parseFloat(getComputedStyle(boot).opacity || '1') > 0.05
            );
            if (bootVisible) {
              sawBoot = true;
              if (!bootBg) bootBg = getComputedStyle(boot).backgroundColor || "";
            }

            const bodyBg = getComputedStyle(document.body).backgroundColor || '';
            const htmlBg = getComputedStyle(document.documentElement).backgroundColor || '';
            const isWhite = (c) => {
              const m = String(c).match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
              if (!m) return false;
              const r = +m[1], g = +m[2], b = +m[3];
              return r > 250 && g > 250 && b > 250;
            };
            const h1 = document.querySelector('[data-testid="stHeading"] h1, h1');
            const input = document.querySelector('[data-testid="stTextInput"] input');
            const hasShell = !!(h1 && input
              && h1.getBoundingClientRect().height > 0
              && input.getBoundingClientRect().height > 0);

            if (!bootVisible && !hasShell) {
              blankFrames += 1;
              if (isWhite(bodyBg) || isWhite(htmlBg)) whiteFrames += 1;
            }

            const deploy = document.querySelector(
              '[data-testid="stAppDeployButton"], .stAppDeployButton'
            );
            if (deploy) {
              const r = deploy.getBoundingClientRect();
              const cs = getComputedStyle(deploy);
              if (
                r.width > 0 && r.height > 0
                && cs.display !== 'none'
                && cs.visibility !== 'hidden'
                && parseFloat(cs.opacity || '1') > 0.05
                && !bootVisible
              ) {
                sawUnstyledDeploy = true;
              }
            }

            for (const sk of document.querySelectorAll(
              '[data-testid="stSkeleton"], .stSkeleton'
            )) {
              const r = sk.getBoundingClientRect();
              const cs = getComputedStyle(sk);
              if (
                r.width > 2 && r.height > 2
                && cs.display !== 'none'
                && cs.visibility !== 'hidden'
                && parseFloat(cs.opacity || '1') > 0.05
                && !bootVisible
              ) {
                sawVisibleSkeleton = true;
                break;
              }
            }

            samples.push({
              t: Math.round(performance.now() - started),
              boot: bootVisible,
              shell: hasShell,
            });

            if (hasShell && (!boot || boot.dataset.done === '1' || boot.hidden)) {
              resolve({
                sawBoot,
                blankFrames,
                whiteFrames,
                sawVisibleSkeleton,
                sawUnstyledDeploy,
                samples: samples.length,
                bootBg,
                readyMs: Math.round(performance.now() - started),
              });
              return;
            }
            if (performance.now() - started >= maxMs) {
              resolve({
                sawBoot,
                blankFrames,
                whiteFrames,
                sawVisibleSkeleton,
                sawUnstyledDeploy,
                samples: samples.length,
                bootBg,
                readyMs: Math.round(performance.now() - started),
                timedOut: true,
              });
              return;
            }
            requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        })""",
        max_ms,
    )


def _chip_y_shift(page, *, settle_ms: int = 2500) -> dict:
    """Track the first starter chip's Y from first paint until settle; report max jump."""
    return page.evaluate(
        """(settleMs) => new Promise((resolve) => {
          const started = performance.now();
          const ys = [];
          let firstY = null;
          let maxShift = 0;
          let prev = null;

          const chipSel = () => {
            const buttons = [...document.querySelectorAll(
              '[class*="st-key-chip_"] button, '
              + '[class*="st-key-chip_"] [data-testid^="stBaseButton"]'
            )];
            return buttons.find((b) => {
              const r = b.getBoundingClientRect();
              const cs = getComputedStyle(b);
              return r.width > 0 && r.height > 0
                && cs.visibility !== 'hidden'
                && parseFloat(cs.opacity || '1') > 0.05;
            }) || null;
          };

          const tick = () => {
            const chip = chipSel();
            if (chip) {
              const y = chip.getBoundingClientRect().top;
              ys.push(y);
              if (firstY === null) firstY = y;
              if (prev !== null) {
                maxShift = Math.max(maxShift, Math.abs(y - prev));
              }
              prev = y;
            }
            if (performance.now() - started >= settleMs) {
              const span = ys.length ? Math.max(...ys) - Math.min(...ys) : 0;
              resolve({
                samples: ys.length,
                maxStepShift: Math.round(maxShift * 10) / 10,
                span: Math.round(span * 10) / 10,
                firstY,
                lastY: prev,
              });
              return;
            }
            requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        })""",
        settle_ms,
    )


@pytest.mark.usefixtures("_production_boot_patch")
def test_first_load_shows_boot_loader_without_unstyled_frames() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            # Start sampling as soon as the document exists.
            page.goto(url, wait_until="commit")
            result = _sample_first_paint(page, max_ms=15000)
            assert result.get("sawBoot"), result
            assert result.get("blankFrames", 99) == 0, result
            assert result.get("whiteFrames", 99) == 0, result
            assert not result.get("sawVisibleSkeleton"), result
            assert not result.get("sawUnstyledDeploy"), result
            assert not result.get("timedOut"), result
            page.get_by_placeholder(READY_HINT).wait_for(state="visible", timeout=30000)
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled(
                timeout=30000
            )
            expect(page.locator(".dk-product")).to_have_text("Ask your document")
            expect(page.locator("#demo-exit")).to_have_attribute(
                "href", "https://roxanatapia.dev/"
            )
            expect(page.locator("#demo-exit")).not_to_have_attribute("target", "_blank")
            expect(page.locator(".dk-exit-visitor")).to_have_text("Back to portfolio")
            expect(page.locator(".dk-exit-visitor")).to_be_visible()
            expect(page.locator(".dk-exit-ops")).to_be_hidden()
            expect(page.get_by_role("link", name="Log out")).to_have_count(0)
            expect(page.locator("footer.dk-footer")).to_be_visible()
            expect(page.get_by_role("link", name="Built by Roxana Tapia")).to_be_visible()
            expect(page.get_by_role("link", name="Book a call")).to_have_attribute(
                "href", _BOOKING_PAGE_URL
            )
            expect(page.get_by_role("link", name="Book a call")).to_have_attribute(
                "target", "_blank"
            )
            expect(page.get_by_role("button", name="Try your own PDF")).to_have_count(0)
            expect(page.locator('[data-testid="stFileUploader"]')).to_have_count(0)
            expect(
                page.get_by_text("NDA · Acme Corp and Beta Labs", exact=False).first
            ).to_be_visible()
            expect(page.get_by_text("sample-nda.pdf")).to_have_count(0)
            browser.close()


@pytest.mark.usefixtures("_production_boot_patch")
def test_first_load_chips_at_390px_do_not_jump() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.goto(url, wait_until="commit")
            paint = _sample_first_paint(page, max_ms=15000)
            assert paint.get("sawBoot"), paint
            assert paint.get("blankFrames", 99) == 0, paint
            assert not paint.get("sawVisibleSkeleton"), paint

            page.get_by_placeholder(READY_HINT).wait_for(state="visible", timeout=30000)
            expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled(
                timeout=30000
            )
            chip = page.get_by_role("button", name=TERM_CHIP, exact=True)
            expect(chip).to_be_visible(timeout=30000)
            header = page.evaluate(
                """() => {
                  const product = document.querySelector('.dk-product');
                  const wordmark = document.querySelector('.dk-wordmark');
                  const productBox = product.getBoundingClientRect();
                  const wordmarkBox = wordmark.getBoundingClientRect();
                  return {
                    text: (product.textContent || '').trim(),
                    overflow: product.scrollWidth - product.clientWidth,
                    whiteSpace: getComputedStyle(product).whiteSpace,
                    productTop: productBox.top,
                    wordmarkBottom: wordmarkBox.bottom,
                  };
                }"""
            )
            assert header["text"] == "Ask your document"
            assert header["whiteSpace"] == "normal"
            assert header["overflow"] <= 1
            assert header["productTop"] >= header["wordmarkBottom"] - 1

            # Reload so we can measure chip motion from first appearance.
            page.goto(url, wait_until="commit")
            shift = _chip_y_shift(page, settle_ms=4000)
            assert shift["samples"] >= 1, shift
            assert shift["maxStepShift"] == 0, shift
            assert shift["span"] == 0, shift
            browser.close()


def _dark_rgb(color: str) -> bool:
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color.strip())
    if not match:
        return False
    red, green, blue = (int(part) for part in match.groups())
    return red == 20 and green == 6 and blue == 31


@pytest.mark.usefixtures("_production_boot_patch")
def test_dark_first_load_has_no_white_flash() -> None:
    """Saved dark theme paints #14061f before the shell, with no white frame."""
    sync_playwright, _expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 1280, "height": 800},
                color_scheme="dark",
            )
            page.add_init_script("localStorage.setItem('theme', 'dark');")
            page.goto(url, wait_until="commit")
            result = _sample_first_paint(page, max_ms=15000)
            assert result.get("sawBoot"), result
            assert result.get("blankFrames", 99) == 0, result
            assert result.get("whiteFrames", 99) == 0, result
            assert _dark_rgb(str(result.get("bootBg") or "")), result
            assert not result.get("timedOut"), result
            theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert theme == "dark"
            browser.close()


@pytest.mark.usefixtures("_production_boot_patch")
def test_operator_header_loads_logout_and_heartbeat() -> None:
    sync_playwright, expect = _import_playwright()
    with _streamlit_server() as url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.set_extra_http_headers({"X-Ops-Ok": "1"})
            page.goto(url, wait_until="commit")
            page.get_by_placeholder(READY_HINT).wait_for(state="visible", timeout=30000)
            assert page.evaluate(
                "document.documentElement.getAttribute('data-operator')"
            ) == "true"
            expect(page.locator("#demo-exit")).to_have_attribute("href", "/ops/logout")
            expect(page.locator(".dk-exit-ops")).to_be_visible()
            expect(page.locator(".dk-exit-visitor")).to_be_hidden()
            expect(page.locator('script[src="/app/demo-kit/ops-heartbeat.js"]')).to_have_count(1)
            expect(page.get_by_role("button", name="Try your own PDF")).to_be_visible()
            expect(page.locator('[data-testid="stFileUploader"]')).to_have_count(0)
            page.get_by_role("button", name="Try your own PDF").click()
            expect(page.locator('[data-testid="stFileUploader"]')).to_be_visible(timeout=30000)
            browser.close()
