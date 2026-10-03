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
    "chat_indexing": "Indexing in progress…",
    "chat_waiting": "Upload a PDF to start asking questions.",
    "doc_ready": (
        "**{name}** is ready. Ask in plain language; "
        "**Sources** opens under the latest answer so you can check the page."
    ),
    "doc_indexing": "Indexing **{name}**…",
    "doc_cleared": "Document cleared. Upload a PDF when you want to continue.",
    "session_fresh": (
        "Upload a PDF below. When the green **ready** message appears, "
        "you can ask your first question."
    ),
    "doc_stale": (
        "This PDF is not indexed yet. Wait for the green **ready** message, "
        "or use the **✕** on the file above and upload again."
    ),
    "chat_blocked_indexing": (
        "Still indexing. Wait for the green **ready** message, then ask again."
    ),
    "chat_blocked_ghost": (
        "The file name may still appear after **Rerun**, but the upload was cleared. "
        "Use the **✕** on the PDF above, or upload again."
    ),
    "chat_blocked_empty": (
        "Upload a PDF and wait until indexing finishes before asking a question."
    ),
    "reindex_resume": "Re-indexing the uploaded PDF…",
    "progress_prepare": "Preparing the PDF…",
    "progress_extract": "Extracting text…",
    "progress_chunk": "Preparing the index…",
    "progress_embed": "Building the search index…",
    "progress_index": "Indexing…",
    "progress_done": "Ready.",
    "toast_indexed": "Document indexed.",
}


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
    """Show the short how-to so a visitor knows to upload, wait, then ask."""
    st.sidebar.markdown(
        """
    <div class="app-sidebar-block">
      <p class="app-sidebar-pitch">
        Private PDF Q&amp;A with <strong>page Sources</strong> you can check.
        One file per session — on infrastructure you control.
      </p>
      <div class="app-sidebar-steps">
        <div class="app-sidebar-step">
          <span class="app-sidebar-step-n" aria-hidden="true">1️⃣</span>
          <span>Upload a PDF</span>
        </div>
        <div class="app-sidebar-step">
          <span class="app-sidebar-step-n" aria-hidden="true">2️⃣</span>
          <span>Wait for <strong>ready</strong></span>
        </div>
        <div class="app-sidebar-step">
          <span class="app-sidebar-step-n" aria-hidden="true">3️⃣</span>
          <span>Ask — Sources opens under the answer</span>
        </div>
      </div>
      <p class="app-sidebar-meta">
        <span aria-hidden="true">💡</span>
        Tip: name a section (e.g. Section 3) when you can.
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
