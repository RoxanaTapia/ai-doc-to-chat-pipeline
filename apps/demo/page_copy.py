"""Visitor sentences and the hero, so the page script can call them."""

from __future__ import annotations

import html
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from ui_theme import inject_theme

# Visitor-facing lines for the pilot. The page script looks them up by key.
_CLIENT_COPY = {
    "hero_title": "Read before you sign",
    "hero_lead": (
        "Beta Labs has an NDA from Acme Corp the night before a kickoff. "
        "Tap a question. Each answer shows the page it came from, and how long it took."
    ),
    "hero_kicker": "",
    "chat_ready": "Ask about this document…",
    "chat_indexing": "Reading the document…",
    "chat_waiting": "The sample NDA is loading…",
    "doc_empty": "Loading the sample NDA…",
    "doc_reading": "Reading {name}: {step}",
    "doc_ready": "NDA · Acme Corp and Beta Labs · {pages} pages",
    "doc_ready_named": "{name} · {pages} pages",
    "doc_answering": "Looking through the document",
    "doc_timeout": "That took too long, so I stopped. Please ask again.",
    "sample_button": "Use sample NDA",
    "sample_help": (
        "Loads the fictional walkthrough NDA in one click. "
        "Do not upload real company files on the shared pilot."
    ),
    "sample_missing": "The sample NDA is not available in this build.",
    "own_pdf_link": "Try your own PDF",
    "own_pdf_help": (
        "Replace the sample with a PDF from your computer. "
        "Do not upload real company files on the shared pilot."
    ),
    "ask_button": "Ask",
    "no_sources": "Searched the document",
    "checked_pages": "Searched all {n} pages",
    "checked_one_page": "Searched the whole page",
    "sample_preload_error": "Could not load the sample NDA.",
    "pdf_unreadable": "We couldn't read this PDF. Try another file.",
    "upload_limit_hint": "PDFs up to {limit}.",
    "upload_too_large": "That PDF is larger than {limit}. Please choose a smaller file.",
    "usage_cap": (
        "This demo has reached its limit for now. "
        "Please try again later, or book a short call and I'll walk you through it."
    ),
    "booking_prompt": "Want this on your documents?",
    "booking_action": "Message me on Upwork.",
    "booking_action_website": "Book a call",
    "progress_prepare": "preparing",
    "progress_extract": "extracting text",
    "progress_chunk": "preparing the index",
    "progress_embed": "building the search index",
    "progress_index": "indexing",
    "progress_done": "ready",
    "chat_blocked_indexing": ("Still reading the document. Wait until you can type, then ask."),
    "chat_blocked_empty": "The sample NDA is still loading. Try again in a moment.",
    "privacy_badge_hosted": "How your data is handled",
    "privacy_hosted": (
        "Your PDF and search stay on this server for this browser session; "
        "only the few passages needed for an answer are sent to a hosted model "
        "({vendor}). Cleared when you leave or reset."
    ),
    "privacy_memory": (
        "Your PDF is kept only for this browser session, and cleared when you leave or reset."
    ),
}

# The sample NDA on disk. Visitors see SAMPLE_DOCUMENT_TITLE, never this filename.
SAMPLE_FILE_NAME = "sample-nda.pdf"
SAMPLE_DOCUMENT_TITLE = "NDA · Acme Corp and Beta Labs"

# Two questions the sample NDA answers; one it does not. Short, so a phone can tap them.
STARTER_CHIPS = (
    {
        "label": "How long does this last?",
        "question": "How long does this last?",
    },
    {
        "label": "Which law applies?",
        "question": "Which law applies?",
    },
    {
        "label": "What's the penalty fee?",
        "question": "What's the penalty fee?",
    },
)

BOOKING_URL_ENV = "DEMO_BOOKING_URL"
# Website path. An Upwork profile URL still uses the Upwork sentence.
WEBSITE_BOOKING_URL = (
    "https://calendar.google.com/calendar/appointments/schedules/"
    "AcZssZ3r5LJ7bPLsMMNRH2fcH9DkzDNTf8oA8XIg0NX9aNpoKezPGMN2Z2OTD-baqG9XoB5vfehzJnzt"
    "?gv=true"
)


# Buyer-facing vendor names. The setting values stay anthropic / openai / xai.
_HOSTED_VENDORS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "xai": "xAI",
}


def privacy_line_for_provider(provider: str) -> str:
    """One accurate privacy line for the active generator. Do not overclaim."""
    name = (provider or "").strip().lower()
    vendor = _HOSTED_VENDORS.get(name)
    if vendor:
        return _CLIENT_COPY["privacy_hosted"].format(vendor=vendor)
    return _CLIENT_COPY["privacy_memory"]


def privacy_badge_for_provider(provider: str) -> str:
    """Collapsed lock-badge label. Do not call a hosted model private."""
    return _CLIENT_COPY["privacy_badge_hosted"]


def _is_upwork_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "upwork.com" or host.endswith(".upwork.com")


def booking_url(raw: str | None = None) -> str:
    """Booking link from env. A blank env uses the website calendar link.

    Pass "" to hide the line. An Upwork profile URL stays an Upwork link.
    """
    if raw is None:
        raw = os.getenv(BOOKING_URL_ENV)
        if not (raw or "").strip():
            return WEBSITE_BOOKING_URL
    return (raw or "").strip()


def booking_cta(url: str | None = None) -> dict[str, str | None]:
    """Copy for the post-third-answer line. ``href`` is None when passed blank.

    The website path says Book a call and opens the calendar link in a new tab.
    An Upwork profile URL keeps "Message me on Upwork."
    """
    href = booking_url(url)
    if not href:
        return {"prompt": "", "action": "", "href": None}
    action_key = "booking_action" if _is_upwork_url(href) else "booking_action_website"
    return {
        "prompt": _CLIENT_COPY["booking_prompt"],
        "action": _CLIENT_COPY[action_key],
        "href": href,
    }


def request_is_operator(headers: Mapping[str, str] | None = None) -> bool:
    """True when the edge sent X-Ops-Ok on this session.

    Presence is enough, including an empty value. The same header turns
    Exit demo into Log out on the document.
    """
    if headers is None:
        headers = st.context.headers
    return any(str(key).lower() == "x-ops-ok" for key in headers)


def human_document_name(file_name: str) -> str:
    """A name a person can read. Sample files keep their story title, not the filename."""
    raw = (file_name or "").strip()
    if raw in {SAMPLE_FILE_NAME, "acme-beta-labs-nda.pdf"}:
        return SAMPLE_DOCUMENT_TITLE
    stem = Path(raw).stem.replace("_", " ").replace("-", " ")
    words = [word for word in stem.split() if word and word.lower() != "pdf"]
    if not words:
        return "Document"
    return " ".join(word[:1].upper() + word[1:] for word in words)


def sample_document_line(pages: int) -> str:
    """The loaded sample card. The page count is this sample PDF."""
    if pages > 0:
        return _CLIENT_COPY["doc_ready"].format(pages=pages)
    return SAMPLE_DOCUMENT_TITLE


def _walkthrough_pairs(messages: list[dict] | None) -> list[tuple[dict | None, dict]]:
    pairs: list[tuple[dict | None, dict]] = []
    pending: dict | None = None
    for message in messages or []:
        if message.get("role") == "user":
            pending = message
            continue
        if message.get("role") == "assistant":
            pairs.append((pending, message))
            pending = None
    return pairs


def walkthrough_tally(messages: list[dict] | None) -> dict[str, int] | None:
    """Counts for the sample end card, or None until all three starter questions have answers."""
    from ai_doc.rag.citations import is_insufficient_answer

    wanted = [chip["question"] for chip in STARTER_CHIPS]
    found: dict[str, dict] = {}
    for user, assistant in _walkthrough_pairs(messages):
        question = ((user or {}).get("content") or "").strip()
        if question in wanted:
            found[question] = assistant
    if any(question not in found for question in wanted):
        return None
    honest = 0
    sources = 0
    for question in wanted:
        assistant = found[question]
        answer = assistant.get("content") or ""
        refusal = bool(assistant.get("refusal")) or is_insufficient_answer(answer, question)
        if refusal:
            honest += 1
        if refusal or assistant.get("sources"):
            sources += 1
    return {"questions": len(wanted), "sources": sources, "honest": honest}


def walkthrough_score(messages: list[dict] | None) -> str | None:
    """End-card line. Numbers are this sample walkthrough, not a general claim."""
    tally = walkthrough_tally(messages)
    if tally is None:
        return None
    return (
        f"{tally['questions']} questions · {tally['sources']} sources · {tally['honest']} honest no"
    )


def _render_case_end(messages: list[dict] | None) -> None:
    """Close the sample story: the tally, a call, and a way out."""
    score = walkthrough_score(messages)
    if not score:
        return
    safe_score = html.escape(score)
    st.markdown(
        (
            '<section class="app-end">'
            '<p class="app-end__kicker">Sample walkthrough</p>'
            f'<p class="app-end__score">{safe_score}</p>'
            '<p class="app-end__actions">'
            f'<a class="app-end__link" href="{html.escape(_PORTFOLIO_URL, quote=True)}">'
            "Book a call</a>"
            '<a class="app-end__link" href="/invite/exit">Exit demo</a>'
            "</p>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def should_show_booking(url: str | None = None) -> bool:
    """Hide the booking line when DEMO_BOOKING_URL is blank."""
    return bool(booking_url(url))


def answered_questions(messages: list[dict] | None) -> set[str]:
    """Questions that already have an assistant reply in this session."""
    answered: set[str] = set()
    pending = ""
    for message in messages or []:
        role = (message or {}).get("role")
        content = ((message or {}).get("content") or "").strip()
        if role == "user":
            pending = content
        elif role == "assistant" and pending:
            answered.add(pending)
            pending = ""
    return answered


def remaining_starter_chips(messages: list[dict] | None) -> list[dict[str, str]]:
    """Starter questions the visitor has not asked yet."""
    done = answered_questions(messages)
    return [chip for chip in STARTER_CHIPS if chip["question"] not in done]


def should_show_starter_chips(
    messages: list[dict] | None,
    query: str | None = None,
    *,
    chat_ready: bool = True,
) -> bool:
    """Starter chips wait until the index is ready, then drop each one as it is used.

    They stay off while the document is still loading, and again while a
    question is in flight, so Streamlit cannot leave greyed chip buttons
    under "Looking through the document…".
    """
    if not chat_ready:
        return False
    if (query or "").strip():
        return False
    return bool(remaining_starter_chips(messages))


def sample_preload_error_copy() -> dict[str, str]:
    """Short error plus the retry button label when sample preload fails."""
    return {
        "error": _CLIENT_COPY["sample_preload_error"],
        "retry": _CLIENT_COPY["sample_button"],
    }


def checked_all_pages_note(pages: int, *, any_blank: bool = False) -> str:
    """Refusal footnote: count pages that produced text; hedge if any were blank."""
    if any_blank or pages <= 0:
        return _CLIENT_COPY["no_sources"]
    if pages == 1:
        return _CLIENT_COPY["checked_one_page"]
    return _CLIENT_COPY["checked_pages"].format(n=pages)


def should_preload_sample(*, own_pdf_mode: bool, has_upload: bool, is_indexed: bool) -> bool:
    """True on first paint when the visitor has not chosen their own PDF."""
    return not own_pdf_mode and not has_upload and not is_indexed


def _inject_demo_styles(*, hide_sidebar: bool) -> None:
    """Load the shared theme and ask-pending client before the hero paints."""
    inject_theme(hide_sidebar=hide_sidebar, busy_label=_CLIENT_COPY["doc_answering"])


def _render_client_hero(*, hide_sidebar: bool = True) -> None:
    """Title plus one subtitle. No setup paragraphs."""
    _inject_demo_styles(hide_sidebar=hide_sidebar)
    st.title(_CLIENT_COPY["hero_title"])
    st.markdown(
        f'<p class="app-hero__lead">{html.escape(_CLIENT_COPY["hero_lead"])}</p>',
        unsafe_allow_html=True,
    )


def _render_privacy_badge(provider: str) -> None:
    """Lock badge. Above 40rem the note overlays; CSS clears the ask box."""
    line = html.escape(privacy_line_for_provider(provider))
    label = html.escape(privacy_badge_for_provider(provider))
    st.markdown(
        f"""
        <details class="app-lock">
          <summary>
            <span class="app-lock__icon" aria-hidden="true">
              <svg viewBox="0 0 16 16" width="12" height="12">
                <path fill="currentColor"
                  d="M8 1.3A3.2 3.2 0 0 0 4.8 4.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1
                  1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.8V4.5A3.2 3.2 0 0 0 8 1.3zm1.6
                  4.7H6.4V4.5a1.6 1.6 0 1 1 3.2 0V6z"/>
              </svg>
            </span>
            {label}
          </summary>
          <p class="app-lock__line">{line}</p>
        </details>
        """,
        unsafe_allow_html=True,
    )


def _render_booking_cta(url: str | None = None) -> None:
    """One next-step line. Hidden entirely when DEMO_BOOKING_URL is blank."""
    if not should_show_booking(url):
        return
    cta = booking_cta(url)
    prompt = html.escape(str(cta["prompt"]))
    action = html.escape(str(cta["action"]))
    href = cta["href"]
    if not href:
        return
    safe_href = html.escape(href, quote=True)
    action_html = (
        f'<a class="app-booking__link" href="{safe_href}" '
        f'target="_blank" rel="noopener noreferrer">{action}</a>'
    )
    st.markdown(
        f'<p class="app-booking">{prompt} {action_html}</p>',
        unsafe_allow_html=True,
    )


def _render_sidebar_pitch() -> None:
    """Developer-only how-to. Client mode has no sidebar."""
    st.sidebar.markdown(
        """
        <div class="app-sidebar-block">
          <p class="app-sidebar-pitch">
            Private PDF Q&amp;A with <strong>page sources</strong> you can check.
            The sample NDA loads on open.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_footer(*, generator_label: str, github_url: str) -> None:
    """Model caption and GitHub sit at the bottom, after developer controls."""
    st.sidebar.markdown(
        f"""
        <div class="app-sidebar-footer">
          <p class="app-sidebar-model">{html.escape(generator_label)}</p>
          <a class="app-sidebar-github"
             href="{html.escape(github_url, quote=True)}"
             target="_blank"
             rel="noopener noreferrer"
             aria-label="Project on GitHub"
             title="Project on GitHub">
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07
                -.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08
                -.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2
                .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
                0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0
                .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


_PORTFOLIO_URL = "https://roxanatapia.dev/"
_CASE_URL = "https://roxanatapia.dev/cases/pdf-chat"
_BOOKING_PAGE_URL = (
    "https://calendar.google.com/calendar/appointments/schedules/"
    "AcZssZ3r5LJ7bPLsMMNRH2fcH9DkzDNTf8oA8XIg0NX9aNpoKezPGMN2Z2OTD-baqG9XoB5vfehzJnzt?gv=true"
)


def client_footer_html(generator_label: str) -> str:
    """Kit footer. How it's built opens the case page. The model line stays quiet."""
    label = html.escape(generator_label)
    portfolio = html.escape(_PORTFOLIO_URL, quote=True)
    case = html.escape(_CASE_URL, quote=True)
    booking = html.escape(_BOOKING_PAGE_URL, quote=True)
    return f"""
<footer class="dk-footer">
  <div class="dk-footer-slot">
    <a href="{portfolio}">Built by Roxana Tapia</a>
    <span aria-hidden="true">·</span>
    <a href="{case}">How it's built</a>
    <span aria-hidden="true">·</span>
    <a href="{booking}" target="_blank" rel="noopener noreferrer">Book a call</a>
    <p class="app-how__line">{label}</p>
  </div>
</footer>
"""


def _render_client_footer(*, generator_label: str) -> None:
    """Kit footer when the sidebar is hidden."""
    st.markdown(client_footer_html(generator_label), unsafe_allow_html=True)
