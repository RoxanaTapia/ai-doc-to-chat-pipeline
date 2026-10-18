"""Visitor sentences and the hero, so the page script can call them."""

from __future__ import annotations

import html
import os

import streamlit as st

from ui_theme import inject_theme

# Visitor-facing lines for the pilot. The page script looks them up by key.
_CLIENT_COPY = {
    "hero_title": "Ask your document",
    "hero_lead": ("Ask your contract anything. Every answer shows the page it came from."),
    "hero_kicker": "",
    "chat_ready": "Ask about this document…",
    "chat_indexing": "Reading the document…",
    "chat_waiting": "The sample NDA is loading…",
    "doc_empty": "Loading the sample NDA…",
    "doc_reading": "Reading {name} — {step}",
    "doc_ready": "Sample NDA · {pages} pages",
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
    "booking_prompt": "Want this on your documents?",
    "booking_action": "Message me on Upwork.",
    "progress_prepare": "preparing",
    "progress_extract": "extracting text",
    "progress_chunk": "preparing the index",
    "progress_embed": "building the search index",
    "progress_index": "indexing",
    "progress_done": "ready",
    "chat_blocked_indexing": ("Still reading the document. Wait until you can type, then ask."),
    "chat_blocked_empty": "The sample NDA is still loading. Try again in a moment.",
    "privacy_badge_ollama": "Runs on this server",
    "privacy_badge_hosted": "How your data is handled",
    "privacy_ollama": (
        "Runs on this server: your PDF, search and answers never leave it. "
        "Kept only for this browser session, and cleared when you leave or reset."
    ),
    "privacy_anthropic": (
        "Your PDF and search stay on this server for this browser session; "
        "only the few passages needed for an answer are sent to a hosted model "
        "(Anthropic). Cleared when you leave or reset."
    ),
    "privacy_memory": (
        "Your PDF is kept only for this browser session, and cleared when you leave or reset."
    ),
}

# Two questions the sample NDA answers; one it does not.
STARTER_CHIPS = (
    {
        "label": "How long does this agreement last?",
        "question": "How long does this agreement last?",
    },
    {
        "label": "Which country's law applies, and where are disputes settled?",
        "question": "Which country's law applies, and where are disputes settled?",
    },
    {
        "label": "What is the penalty fee if Beta Labs breaks this agreement?",
        "question": "What is the penalty fee if Beta Labs breaks this agreement?",
    },
)

BOOKING_URL_ENV = "DEMO_BOOKING_URL"


def privacy_line_for_provider(provider: str) -> str:
    """One accurate privacy line for the active generator. Do not overclaim."""
    name = (provider or "").strip().lower()
    if name == "ollama":
        return _CLIENT_COPY["privacy_ollama"]
    if name == "anthropic":
        return _CLIENT_COPY["privacy_anthropic"]
    return _CLIENT_COPY["privacy_memory"]


def privacy_badge_for_provider(provider: str) -> str:
    """Collapsed lock-badge label. Do not call a hosted model private."""
    name = (provider or "").strip().lower()
    if name == "ollama":
        return _CLIENT_COPY["privacy_badge_ollama"]
    return _CLIENT_COPY["privacy_badge_hosted"]


def booking_url(raw: str | None = None) -> str:
    """Upwork profile from env, or empty. Blank hides the whole line."""
    if raw is None:
        raw = os.getenv(BOOKING_URL_ENV)
    return (raw or "").strip()


def booking_cta(url: str | None = None) -> dict[str, str | None]:
    """Copy for the post-third-answer line. ``href`` is None when the env is blank."""
    href = booking_url(url)
    if not href:
        return {"prompt": "", "action": "", "href": None}
    return {
        "prompt": _CLIENT_COPY["booking_prompt"],
        "action": _CLIENT_COPY["booking_action"],
        "href": href,
    }


def should_show_booking(url: str | None = None) -> bool:
    """Hide the booking line when DEMO_BOOKING_URL is blank."""
    return bool(booking_url(url))


def should_show_starter_chips(
    messages: list[dict] | None,
    query: str | None = None,
    *,
    chat_ready: bool = True,
) -> bool:
    """Starter chips wait until the index is ready, then hide after the first ask.

    They stay off while the document is still loading, and again while a
    question is in flight, so Streamlit cannot leave greyed chip buttons
    under "Looking through the document…".
    """
    if not chat_ready:
        return False
    if (query or "").strip():
        return False
    return not any((message or {}).get("role") == "assistant" for message in (messages or []))


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
    """Load the shared theme before the hero paints."""
    inject_theme(hide_sidebar=hide_sidebar)


def _render_client_hero(*, hide_sidebar: bool = True) -> None:
    """Title plus one subtitle. No setup paragraphs."""
    _inject_demo_styles(hide_sidebar=hide_sidebar)
    st.title(_CLIENT_COPY["hero_title"])
    st.markdown(
        f'<p class="app-hero__lead">{html.escape(_CLIENT_COPY["hero_lead"])}</p>',
        unsafe_allow_html=True,
    )


def _render_privacy_badge(provider: str) -> None:
    """Lock badge that expands to the model-dependent privacy line."""
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


def _render_client_footer(*, generator_label: str, github_url: str) -> None:
    """Quiet exit + model line when the sidebar is hidden."""
    st.markdown(
        f"""
        <div class="app-footer">
          <a class="app-footer__link" href="/invite/exit">Exit demo</a>
          <span class="app-footer__dot" aria-hidden="true">·</span>
          <a class="app-footer__link"
             href="{html.escape(github_url, quote=True)}"
             target="_blank"
             rel="noopener noreferrer">GitHub</a>
          <span class="app-footer__dot" aria-hidden="true">·</span>
          <span class="app-footer__meta">{html.escape(generator_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
