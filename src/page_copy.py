"""Visitor sentences and the hero, so the page script can call them."""

from __future__ import annotations

import streamlit as st

from ui_theme import inject_theme

# Visitor-facing lines for the pilot. The page script looks them up by key.
_CLIENT_COPY = {
    "hero_title": "Ask your document",
    "hero_lead": (
        "Grounded answers from a confidential PDF, with "
        "<strong>page-level sources</strong> you can verify. "
        "Runs on infrastructure you control."
    ),
    "hero_kicker": "Policies, SOPs, reports, contracts — upload one PDF, then ask.",
    "chat_ready": "Ask about this document…",
    "chat_indexing": "Reading the document…",
    "chat_waiting": "Upload a PDF or try the sample NDA.",
    "doc_empty": "Upload a PDF, or try the sample NDA.",
    "doc_reading": "Reading {name} — {step}",
    "doc_ready": "{name} is ready. Ask a question.",
    "doc_answering": "Answering…",
    "sample_button": "Try the sample NDA",
    "sample_help": (
        "Loads the fictional walkthrough NDA in one click. "
        "Do not upload real company files on the shared pilot."
    ),
    "sample_missing": "The sample NDA is not available in this build.",
    "ask_button": "Ask",
    "no_sources": "No sources used.",
    "progress_prepare": "preparing",
    "progress_extract": "extracting text",
    "progress_chunk": "preparing the index",
    "progress_embed": "building the search index",
    "progress_index": "indexing",
    "progress_done": "ready",
    "chat_blocked_indexing": (
        "Still reading the document. Wait until the status says ready, then ask."
    ),
    "chat_blocked_empty": (
        "Upload a PDF or try the sample NDA, then wait until it is ready."
    ),
}

# Two questions the sample NDA answers; one it does not.
STARTER_CHIPS = (
    {
        "label": "What's confidential information?",
        "question": "How is Confidential Information defined in Section 1?",
    },
    {
        "label": "What does Section 3 require?",
        "question": "What obligations does Section 3 impose on the Receiving Party?",
    },
    {
        "label": "Any liquidated damages?",
        "question": "Does this agreement specify liquidated damages?",
    },
)


def _inject_demo_styles() -> None:
    """Load the shared theme before the hero paints."""
    inject_theme()


def _render_client_hero() -> None:
    """Open the page on the document title so the visitor knows what to do."""
    _inject_demo_styles()
    st.title(_CLIENT_COPY["hero_title"])
    st.markdown(
        f"""
        <div class="app-hero">
          <p class="app-hero__lead">{_CLIENT_COPY["hero_lead"]}</p>
          <p class="app-hero__kicker">{_CLIENT_COPY["hero_kicker"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_pitch() -> None:
    """Show the short how-to in plain text at the top of the sidebar."""
    st.sidebar.markdown(
        """
        <div class="app-sidebar-block">
          <p class="app-sidebar-pitch">
            Private PDF Q&amp;A with <strong>page sources</strong> you can check.
            One file per session, on infrastructure you control.
          </p>
          <p class="app-sidebar-howto">
            Upload a PDF, or try the sample NDA. When the status says ready,
            type a question and press Ask. A page badge appears under the
            answer so you can verify the clause.
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
          <p class="app-sidebar-model">{generator_label}</p>
          <a class="app-sidebar-github"
             href="{github_url}"
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
