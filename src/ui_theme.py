"""Streamlit-safe theme for the answer-first client demo.

Off-white paper, near-black ink, one indigo accent, rounded cards, no sidebar.
"""

from __future__ import annotations

import streamlit as st

_THEME_CSS = """
:root {
  --app-bg: #f7f7f5;
  --app-ink: #1a1f24;
  --app-slate: #5c6670;
  --app-muted: #7a838c;
  --app-line: #e6e8ec;
  --app-panel: #ffffff;
  --app-paper: #f7f7f5;
  --app-accent: #3d4fd8;
  --app-accent-hover: #4a5be0;
  --app-accent-ink: #ffffff;
  --app-accent-soft: #eef0fc;
  --app-warn: #6b5330;
  --app-warn-bg: #fbf6ee;
  --app-err: #8a1f2b;
  --app-err-bg: #fdecef;
  --app-ok: #2f6b4f;
  --app-ok-bg: #eaf6ef;
  --app-radius: 16px;
  --app-space: 8px;
  --app-shadow: 0 1px 2px rgba(26, 31, 36, 0.04), 0 12px 32px rgba(26, 31, 36, 0.05);
}

html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}

.stApp {
  background: var(--app-bg);
}

section.main .block-container {
  padding-top: 3.25rem;
  padding-bottom: 5rem;
  max-width: 860px;
}

section.main h1 {
  color: var(--app-ink);
  font-weight: 650;
  letter-spacing: -0.035em;
  line-height: 1.15;
  margin-bottom: 0.45rem;
}

.app-hero__lead {
  font-size: 1.12rem;
  line-height: 1.5;
  color: var(--app-slate);
  margin: 0 0 1.35rem 0;
  max-width: 34rem;
}

.app-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin: 0 0 1.75rem 0;
}

/* Outermost Streamlit block wraps the page; never inherit card or button :has() styles. */
section.main .block-container > div[data-testid="stVerticalBlock"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.app-lock {
  display: inline-block;
  max-width: 100%;
}

.app-lock summary {
  list-style: none;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  cursor: pointer;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  border: 1px solid var(--app-line);
  background: var(--app-panel);
  color: var(--app-slate);
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.app-lock summary::-webkit-details-marker {
  display: none;
}

.app-lock__icon {
  display: inline-flex;
  color: var(--app-accent);
}

.app-lock[open] {
  display: block;
  width: 100%;
}

.app-lock[open] summary {
  border-color: color-mix(in srgb, var(--app-accent) 35%, var(--app-line));
  color: var(--app-ink);
}

.app-lock__line {
  margin: 0.45rem 0 0 0;
  max-width: none;
  width: 100%;
  font-size: 0.9rem;
  line-height: 1.45;
  color: var(--app-slate);
}

[data-testid="stHorizontalBlock"]:has(.app-lock) {
  flex-wrap: wrap !important;
  overflow: visible !important;
}

[data-testid="column"]:has(.app-lock),
[data-testid="stColumn"]:has(.app-lock) {
  overflow: visible !important;
}

[data-testid="column"]:has(.app-lock[open]),
[data-testid="stColumn"]:has(.app-lock[open]) {
  min-width: 100% !important;
  flex: 1 1 100% !important;
  width: 100% !important;
}

.app-status {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space);
  font-size: 0.9rem;
  line-height: 1.4;
  margin: 0;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--app-muted);
}

.app-status--reading,
.app-status--answering {
  color: var(--app-accent);
}

.app-status--ready {
  color: var(--app-muted);
}

.app-status--error {
  color: var(--app-err);
}

.app-own-pdf {
  font-size: 0.9rem;
}

[data-testid="column"]:has(.app-own-pdf) button,
[data-testid="stColumn"]:has(.app-own-pdf) button {
  background: none !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--app-accent) !important;
  font-weight: 600 !important;
  padding: 0 !important;
  min-height: 0 !important;
}

.app-ask-mark {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}

div[data-testid="stVerticalBlock"]:has(.app-ask-mark) [data-testid="stTextInput"] input {
  font-size: 1.05rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
}

/* Streamlit paints this while the field is dirty (uncommitted). Hide it; Enter
   still commits, and Ask/Enter both submit via the question-box callback. */
div[data-testid="stVerticalBlock"]:has(.app-ask-mark) [data-testid="InputInstructions"] {
  display: none !important;
}

[class*="st-key-ask_submit"] button,
[class*="st-key-ask_submit"] [data-testid^="stBaseButton"] {
  background: var(--app-accent) !important;
  background-color: var(--app-accent) !important;
  color: var(--app-accent-ink) !important;
  border: none !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  font-weight: 650 !important;
  padding: 0.55rem 1.25rem !important;
  min-height: 2.6rem !important;
}

[class*="st-key-ask_submit"] button:hover,
[class*="st-key-ask_submit"] [data-testid^="stBaseButton"]:hover {
  background: var(--app-accent-hover) !important;
  background-color: var(--app-accent-hover) !important;
  color: var(--app-accent-ink) !important;
}

.app-chips-label {
  font-size: 0.9rem;
  color: var(--app-muted);
  margin: 0.15rem 0 0.35rem 0;
}

[class*="st-key-chip_"] button,
[class*="st-key-chip_"] [data-testid^="stBaseButton"] {
  border: 1px solid var(--app-line) !important;
  background: var(--app-panel) !important;
  background-color: var(--app-panel) !important;
  color: var(--app-ink) !important;
  border-radius: 999px !important;
  font-weight: 550 !important;
  box-shadow: none !important;
  padding: 0.4rem 0.95rem !important;
  min-height: 2.25rem !important;
  width: auto !important;
}

[class*="st-key-chip_"] button:hover,
[class*="st-key-chip_"] [data-testid^="stBaseButton"]:hover {
  border-color: var(--app-accent) !important;
  color: var(--app-accent) !important;
  background: var(--app-accent-soft) !important;
  background-color: var(--app-accent-soft) !important;
}

.app-typing {
  display: inline-flex;
  align-items: baseline;
  gap: 0.15rem;
  margin: 0.35rem 0 1rem 0;
  color: var(--app-slate);
  font-size: 1rem;
}

.app-typing-dots::after {
  display: inline-block;
  width: 1.2em;
  content: "";
  animation: app-dots 1.2s steps(4, end) infinite;
}

@keyframes app-dots {
  0% { content: ""; }
  25% { content: "."; }
  50% { content: ".."; }
  75% { content: "..."; }
}

[class*="st-key-answer_card_"],
[class*="st-key-refusal_card_"] {
  background: var(--app-panel);
  border: 1px solid var(--app-line);
  border-radius: var(--app-radius);
  box-shadow: var(--app-shadow);
  padding: 1.15rem 1.35rem 1.2rem 1.35rem;
  margin: 0 0 1.1rem 0;
}

[class*="st-key-refusal_card_"] {
  background: var(--app-warn-bg);
  border-color: color-mix(in srgb, var(--app-warn) 18%, var(--app-line));
  box-shadow: none;
}

.app-answer-q {
  margin: 0 0 0.55rem 0;
  font-size: 0.9rem;
  line-height: 1.35;
  color: var(--app-muted);
}

.app-answer-headline {
  margin: 0 0 0.35rem 0;
  font-size: 2rem;
  font-weight: 650;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: var(--app-ink);
}

[class*="st-key-answer_card_"] .app-answer-headline,
[class*="st-key-answer_card_"] [data-testid="stMarkdownContainer"] .app-answer-headline {
  font-size: 2rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: var(--app-ink) !important;
}

.app-answer-body {
  margin: 0 0 0.65rem 0;
  font-size: 1rem;
  font-weight: 400;
  line-height: 1.5;
  color: var(--app-slate);
}

[class*="st-key-refusal_card_"] .app-answer-headline,
[class*="st-key-refusal_card_"] [data-testid="stMarkdownContainer"] .app-answer-headline {
  font-size: 1.2rem !important;
  letter-spacing: -0.02em;
  font-weight: 600 !important;
  color: var(--app-warn) !important;
}

.app-page-badge {
  display: inline-flex;
  align-items: center;
  margin: 0.35rem 0 0.2rem 0;
  padding: 2px 8px;
  border-radius: 8px;
  background: var(--app-accent-soft);
  color: var(--app-accent);
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.app-source-quote,
.app-inline-quote {
  margin: 0 0 0.35rem 0;
  padding: 0.25rem 0 0.1rem 0.85rem;
  border-left: 2px solid color-mix(in srgb, var(--app-accent) 45%, var(--app-line));
  color: var(--app-slate);
  font-size: 0.92rem;
  font-weight: 400;
  line-height: 1.5;
}

button.app-quote-btn {
  display: block;
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  padding: 0;
  cursor: pointer;
}

.app-no-sources {
  margin: 0.45rem 0 0 0;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--app-muted);
}

.app-timing {
  margin: 0.4rem 0 0 0;
  font-size: 0.9rem;
  line-height: 1.3;
  font-weight: 400;
  color: var(--app-muted);
}

.app-booking {
  margin: 0.15rem 0 1.35rem 0;
  font-size: 0.95rem;
  color: var(--app-slate);
}

.app-booking__link {
  color: var(--app-accent);
  font-weight: 600;
  text-decoration: none;
}

.app-booking__link:hover {
  text-decoration: underline;
}

.app-footer {
  margin-top: 2.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--app-line);
  font-size: 0.9rem;
  color: var(--app-muted);
}

.app-footer__link {
  color: var(--app-muted);
  text-decoration: none;
}

.app-footer__link:hover {
  color: var(--app-ink);
}

.app-footer__dot {
  margin: 0 0.4rem;
}

.app-page-preview {
  margin: 0.35rem 0 0.75rem 0;
  border: 1px solid var(--app-line);
  border-radius: 12px;
  overflow: hidden;
}

[data-testid="stChatAvatar"],
[data-testid="chatAvatar"],
[data-testid="stChatMessageAvatar"] {
  display: none !important;
}

.app-source-item {
  margin: 0 0 0.85rem 0;
  padding: 0 0 0.75rem 0;
  border-bottom: 1px solid var(--app-line);
}

.app-source-page {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--app-ink);
}

[data-testid="stCaptionContainer"] {
  color: var(--app-muted);
}

[data-testid="stSidebar"] {
  background: var(--app-panel);
}

[data-testid="stSidebar"] .app-sidebar-block {
  margin: 0.15rem 0 0.85rem 0;
  padding: 0.85rem 0.9rem;
  border-radius: var(--app-radius);
  background: var(--app-bg);
  border: 1px solid var(--app-line);
}

[data-testid="stSidebar"] .app-sidebar-pitch {
  margin: 0;
  font-size: 0.92rem;
  line-height: 1.45;
  color: var(--app-slate);
}

[data-testid="stSidebar"] .app-sidebar-footer {
  margin-top: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

[data-testid="stSidebar"] .app-sidebar-model {
  margin: 0;
  font-size: 0.9rem;
  color: var(--app-muted);
}

[data-testid="stSidebar"] .app-sidebar-github {
  display: inline-flex;
  color: var(--app-muted);
  text-decoration: none;
  padding: 0.35rem;
  border-radius: 0.35rem;
}

.stAppDeployButton,
[data-testid="stAppDeployButton"],
[data-testid="stToolbar"] {
  display: none !important;
}

#MainMenu {
  visibility: hidden;
}

footer {
  visibility: hidden;
}
"""

_HIDE_SIDEBAR_CSS = """
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] {
  display: none !important;
}
"""


def inject_theme(*, hide_sidebar: bool = True) -> None:
    """Inject demo theme CSS once per render (Streamlit-safe)."""
    css = _THEME_CSS + (_HIDE_SIDEBAR_CSS if hide_sidebar else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
