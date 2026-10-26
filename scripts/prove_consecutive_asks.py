#!/usr/bin/env python3
"""Drive a live Streamlit demo in Chromium.

CI does not install Playwright. Run this locally (or in a VM with browsers):

    python3 scripts/prove_consecutive_asks.py --mode hang --start-server
    python3 scripts/prove_consecutive_asks.py --mode pass --start-server

Hang mode is for main (write_stream still present). Pass mode is for the fix.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
APP_PATH = PROJECT_ROOT / "apps" / "demo" / "app.py"

TERM_QUESTION = "How long does this agreement last?"
FEE_QUESTION = "What is the penalty fee if Beta Labs breaks this agreement?"
LAW_QUESTION = "Which country's law applies, and where are disputes settled?"
BOOKING_PROMPT = "Want this on your documents?"
BOOKING_ACTION = "Message me on Upwork."
LOOKING = "Looking through the document"
READY_HINT = "Ask about this document"

STUB_ENV = {
    "USE_STUB_EMBEDDINGS": "1",
    "DEMO_STUB_LLM": "1",
    "USE_DUMMY_GENERATOR": "1",
    "APP_PRESENTATION_MODE": "client",
    "TOKENIZERS_PARALLELISM": "false",
    "DEMO_BOOKING_URL": "https://www.upwork.com/freelancers/roxanadev",
}


def _import_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is not installed. CI skips this script. Locally: "
            "pip install playwright && python3 -m playwright install chromium"
        ) from exc
    return sync_playwright, PlaywrightTimeout


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
        time.sleep(0.4)
    raise SystemExit(f"Streamlit did not become ready at {url}: {last_error}")


@contextmanager
def _reranker_off() -> None:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    patched = original.replace(
        "    reranker:\n      enabled: true",
        "    reranker:\n      enabled: false",
        1,
    )
    if patched == original:
        yield
        return
    CONFIG_PATH.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        CONFIG_PATH.write_text(original, encoding="utf-8")


def _server_env(delay_ms: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(STUB_ENV)
    env["DEMO_STUB_STREAM_DELAY_MS"] = str(delay_ms)
    env.pop("LLM_PROVIDER", None)
    env.pop("API_BASE_URL", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SRC_DIR)
    )
    return env


def _start_streamlit(port: int, delay_ms: int) -> subprocess.Popen:
    env = _server_env(delay_ms)
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
    try:
        _wait_http(f"http://127.0.0.1:{port}/app", timeout=90)
    except SystemExit:
        _stop_streamlit(proc)
        raise
    return proc


def _stop_streamlit(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
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


def _card_questions(page) -> list[str]:
    return [
        (node.text_content() or "").strip()
        for node in page.locator(".app-answer-q").all()
        if (node.text_content() or "").strip()
    ]


def _finished_card_count(page) -> int:
    answers = page.locator('[class*="st-key-answer_card_"]').count()
    refusals = page.locator('[class*="st-key-refusal_card_"]').count()
    return answers + refusals


def _page_text(page) -> str:
    return page.locator("body").inner_text()


def _looking_visible(page) -> bool:
    return LOOKING in _page_text(page)


def _booking_visible(page) -> bool:
    body = _page_text(page)
    return BOOKING_PROMPT in body and BOOKING_ACTION in body


def _screenshot(page, out_dir: Path, name: str) -> str:
    path = out_dir / name
    page.screenshot(path=str(path), full_page=True)
    return str(path)


def _wait_ready(page, timeout_ms: int) -> None:
    from playwright.sync_api import expect

    box = page.get_by_placeholder(READY_HINT)
    box.wait_for(state="visible", timeout=timeout_ms)
    expect(page.get_by_role("button", name="Ask", exact=True)).to_be_enabled(
        timeout=timeout_ms
    )


def _ask(
    page,
    question: str,
    *,
    expect_text: str,
    min_cards: int,
    card_timeout_ms: int,
    out_dir: Path | None = None,
    looking_shot: str = "",
) -> dict:
    """Wait until looking-through ends and a finished card contains expect_text."""
    from playwright.sync_api import expect

    ask = page.get_by_role("button", name="Ask", exact=True)
    expect(ask).to_be_enabled(timeout=card_timeout_ms)
    box = page.get_by_placeholder(READY_HINT)
    box.click()
    box.fill(question)
    expect(ask).to_be_enabled(timeout=card_timeout_ms)
    started = time.perf_counter()
    page.get_by_role("button", name="Ask", exact=True).click()
    looking_seen = False
    deadline = time.time() + (card_timeout_ms / 1000)
    while time.time() < deadline:
        if _looking_visible(page):
            if not looking_seen and out_dir and looking_shot:
                _screenshot(page, out_dir, looking_shot)
            looking_seen = True
            time.sleep(0.2)
            continue
        body = _page_text(page)
        cards = _card_questions(page)
        finished = _finished_card_count(page)
        if expect_text in body and question in cards and finished >= min_cards:
            return {
                "ok": True,
                "seconds": round(time.perf_counter() - started, 2),
                "looking_seen": looking_seen,
                "looking_now": False,
                "cards": cards,
                "finished_cards": finished,
            }
        time.sleep(0.2)
    body = _page_text(page)
    return {
        "ok": False,
        "seconds": round(time.perf_counter() - started, 2),
        "looking_seen": looking_seen or _looking_visible(page),
        "looking_now": _looking_visible(page),
        "cards": _card_questions(page),
        "finished_cards": _finished_card_count(page),
        "body_excerpt": body[:1500],
    }


def _run_pass(page, *, out_dir: Path, card_timeout_ms: int) -> dict:
    shots = []
    q1 = _ask(
        page,
        TERM_QUESTION,
        expect_text="3 years",
        min_cards=1,
        card_timeout_ms=card_timeout_ms,
        out_dir=out_dir,
        looking_shot="pass-q1-looking-through.png",
    )
    looking_path = out_dir / "pass-q1-looking-through.png"
    if looking_path.is_file():
        shots.append(str(looking_path))
    shots.append(_screenshot(page, out_dir, "pass-q1-term.png"))
    q2 = _ask(
        page,
        FEE_QUESTION,
        expect_text="isn't in the document",
        min_cards=2,
        card_timeout_ms=card_timeout_ms,
    )
    shots.append(_screenshot(page, out_dir, "pass-q2-refusal.png"))
    q3 = _ask(
        page,
        LAW_QUESTION,
        expect_text="Spain / Madrid",
        min_cards=3,
        card_timeout_ms=card_timeout_ms,
    )
    shots.append(_screenshot(page, out_dir, "pass-q3-law-booking.png"))
    cards = _card_questions(page)
    body = _page_text(page)
    ok = (
        q1["ok"]
        and q2["ok"]
        and q3["ok"]
        and TERM_QUESTION in cards
        and FEE_QUESTION in cards
        and LAW_QUESTION in cards
        and "3 years" in body
        and "isn't in the document" in body
        and "Spain / Madrid" in body
        and _booking_visible(page)
        and _finished_card_count(page) >= 3
    )
    return {
        "mode": "pass",
        "ok": ok,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "cards": cards,
        "finished_cards": _finished_card_count(page),
        "booking": _booking_visible(page),
        "looking_while_q1": q1.get("looking_seen"),
        "looking_while_q2": q2.get("looking_seen"),
        "looking_while_q3": q3.get("looking_seen"),
        "screenshots": shots,
    }


def _run_hang(page, *, out_dir: Path, card_timeout_ms: int, hang_wait: int) -> dict:
    shots = []
    q1 = _ask(
        page,
        TERM_QUESTION,
        expect_text="3 years",
        min_cards=1,
        card_timeout_ms=card_timeout_ms,
    )
    shots.append(_screenshot(page, out_dir, "hang-q1-term.png"))
    if not q1["ok"]:
        return {
            "mode": "hang",
            "ok": False,
            "reproduced_hang": False,
            "q1": q1,
            "cards": _card_questions(page),
            "screenshots": shots,
            "error": "first question did not finish, so the second-ask hang was not tested",
        }
    q2 = _ask(
        page,
        LAW_QUESTION,
        expect_text="Spain / Madrid",
        min_cards=2,
        card_timeout_ms=hang_wait * 1000,
    )
    shots.append(_screenshot(page, out_dir, "hang-q2-after-wait.png"))
    hung = (not q2["ok"]) and q2.get("finished_cards", 0) < 2
    return {
        "mode": "hang",
        "ok": hung,
        "reproduced_hang": hung,
        "q1": q1,
        "q2": q2,
        "cards": _card_questions(page),
        "finished_cards": _finished_card_count(page),
        "looking_after_wait": _looking_visible(page),
        "hang_wait_s": hang_wait,
        "screenshots": shots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hang", "pass"), required=True)
    parser.add_argument("--url", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=80)
    parser.add_argument("--hang-wait", type=int, default=75)
    parser.add_argument("--card-timeout", type=int, default=60)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    sync_playwright, PlaywrightTimeout = _import_playwright()
    out_dir = Path(args.out_dir) if args.out_dir else Path("/tmp/prove-consecutive-asks")
    out_dir.mkdir(parents=True, exist_ok=True)

    port = args.port or _free_port()
    url = args.url or f"http://127.0.0.1:{port}/app"
    proc = None
    result: dict = {"mode": args.mode, "url": url}

    def _finish(payload: dict, code: int) -> int:
        result.update(payload)
        printed = json.dumps(result, indent=2)
        print(printed)
        (out_dir / f"{args.mode}-result.json").write_text(printed + "\n", encoding="utf-8")
        return code

    try:
        with _reranker_off():
            if args.start_server:
                proc = _start_streamlit(port, args.delay_ms)
                result["server_pid"] = proc.pid
            else:
                _wait_http(url, timeout=20)

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1100, "height": 1400})
                page.set_default_timeout(args.card_timeout * 1000)
                page.goto(url, wait_until="domcontentloaded")
                _wait_ready(page, timeout_ms=90_000)
                _screenshot(page, out_dir, f"{args.mode}-ready.png")
                if args.mode == "hang":
                    payload = _run_hang(
                        page,
                        out_dir=out_dir,
                        card_timeout_ms=args.card_timeout * 1000,
                        hang_wait=args.hang_wait,
                    )
                else:
                    payload = _run_pass(
                        page,
                        out_dir=out_dir,
                        card_timeout_ms=args.card_timeout * 1000,
                    )
                browser.close()
        return _finish(payload, 0 if payload.get("ok") else 1)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return _finish(result, 2)
    finally:
        _stop_streamlit(proc)


if __name__ == "__main__":
    raise SystemExit(main())
