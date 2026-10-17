"""Streamlit-safe theme tokens aligned with receipt-intelligence-demo.

Off-white paper, near-black ink, one indigo accent, 12px cards, 8px spacing.
"""

from __future__ import annotations

import streamlit as st

_THEME_CSS = """
:root {
  --app-bg: #f6f7f9;
  --app-ink: #1a1f24;
  --app-slate: #5c6670;
  --app-muted: #5c6670;
  --app-line: #e6e8ec;
  --app-panel: #ffffff;
  --app-paper: #f6f7f9;
  --app-accent: #3d4fd8;
  --app-accent-hover: #4a5be0;
  --app-accent-ink: #ffffff;
  --app-accent-soft: #eef0fc;
  --app-warn: #8a4b00;
  --app-warn-bg: #fff6e8;
  --app-err: #8a1f2b;
  --app-err-bg: #fdecef;
  --app-ok: #2f6b4f;
  --app-ok-bg: #eaf6ef;
  --app-radius: 12px;
  --app-space: 8px;
  --app-shadow: 0 1px 2px rgba(26, 31, 36, 0.04), 0 10px 28px rgba(26, 31, 36, 0.06);
}

.stApp {
  background:
    radial-gradient(ellipse 70% 45% at 0% 0%, rgba(61, 79, 216, 0.06) 0%, transparent 55%),
    radial-gradient(ellipse 50% 35% at 100% 10%, rgba(196, 154, 90, 0.07) 0%, transparent 50%),
    var(--app-bg);
}

section.main .block-container {
  padding-top: 2rem;
  padding-bottom: 4rem;
  max-width: 880px;
}

section.main h1 {
  color: var(--app-ink);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  margin-bottom: 0.35rem;
}

.app-hero {
  margin: 0 0 1.25rem 0;
  padding: 0 0 1rem 0;
}

.app-hero__lead {
  font-size: 1.1rem;
  line-height: 1.5;
  color: var(--app-slate);
  margin: 0.15rem 0 0.5rem 0;
}

.app-hero__kicker {
  font-size: 0.95rem;
  line-height: 1.45;
  color: var(--app-muted);
  margin: 0;
}

.app-status {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space);
  font-size: 0.95rem;
  line-height: 1.4;
  margin: 0 0 1rem 0;
  padding: 4px 12px;
  border-radius: var(--app-radius);
  color: var(--app-slate);
  background: var(--app-panel);
  border: 1px solid var(--app-line);
}

.app-status--empty {
  color: var(--app-muted);
}

.app-status--reading,
.app-status--answering {
  color: var(--app-accent);
  background: var(--app-accent-soft);
  border-color: transparent;
}

.app-status--ready {
  color: var(--app-ok);
  background: var(--app-ok-bg);
  border-color: transparent;
}

.app-status--error {
  color: var(--app-err);
  background: var(--app-err-bg);
  border-color: transparent;
}

/* Primary action: upload + sample sit in one obvious card */
div[data-testid="stVerticalBlock"]:has(.app-primary-mark) {
  background: var(--app-panel);
  border: 1px solid var(--app-line);
  border-radius: var(--app-radius);
  box-shadow: var(--app-shadow);
  padding: 16px 20px 12px 20px;
  margin: 0 0 1rem 0;
}

.app-primary-mark {
  font-size: 0.8rem;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--app-accent);
  margin: 0 0 0.35rem 0;
}

.app-privacy {
  margin: 0.55rem 0 0.1rem 0;
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--app-muted);
}

/* Answer cards: grey question, larger semibold answer */
div[data-testid="stVerticalBlock"]:has(.app-answer-q) {
  background: var(--app-panel);
  border: 1px solid var(--app-line);
  border-radius: var(--app-radius);
  box-shadow: var(--app-shadow);
  padding: 14px 18px 16px 18px;
  margin: 0 0 0.85rem 0;
}

div[data-testid="stVerticalBlock"]:has(.app-answer-card--refusal) {
  background: var(--app-warn-bg);
  border-color: color-mix(in srgb, var(--app-warn) 22%, var(--app-line));
  box-shadow: none;
}

.app-answer-q {
  margin: 0 0 0.35rem 0;
  font-size: 0.82rem;
  line-height: 1.35;
  color: var(--app-muted);
}

.app-answer-a,
div[data-testid="stVerticalBlock"]:has(.app-answer-q) [data-testid="stMarkdownContainer"] p,
div[data-testid="stVerticalBlock"]:has(.app-answer-q) [data-testid="stMarkdownContainer"] li {
  font-size: 1.15rem;
  font-weight: 600;
  line-height: 1.4;
  color: var(--app-ink);
}

div[data-testid="stVerticalBlock"]:has(.app-answer-card--refusal)
  [data-testid="stMarkdownContainer"] p {
  font-weight: 600;
  color: var(--app-warn);
}

.app-page-badge {
  display: inline-flex;
  align-items: center;
  margin: 0.55rem 0 0.25rem 0;
  padding: 2px 8px;
  border-radius: 8px;
  background: var(--app-accent-soft);
  color: var(--app-accent);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.app-source-quote,
.app-inline-quote {
  margin: 0 0 0.55rem 0;
  padding: 0.35rem 0 0.15rem 0.85rem;
  border-left: 2px solid color-mix(in srgb, var(--app-accent) 45%, var(--app-line));
  color: var(--app-slate);
  font-size: 0.9rem;
  font-weight: 400;
  line-height: 1.5;
}

.app-no-sources {
  margin: 0.45rem 0 0 0;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--app-muted);
}

.app-timing {
  margin: 0.4rem 0 0 0;
  font-size: 0.72rem;
  line-height: 1.3;
  font-weight: 400;
  color: var(--app-muted);
}

[data-testid="stChatAvatar"],
[data-testid="chatAvatar"],
[data-testid="stChatMessageAvatar"] {
  display: none !important;
}

.app-chips-label {
  font-size: 0.85rem;
  color: var(--app-muted);
  margin: 0.25rem 0 0.15rem 0;
}

/* Chips: pill buttons */
div[data-testid="stHorizontalBlock"]:has(.app-chip-mark) button {
  border: 1px solid var(--app-line);
  background: var(--app-panel);
  color: var(--app-ink);
  border-radius: 999px;
  font-weight: 600;
  box-shadow: none;
}

div[data-testid="stHorizontalBlock"]:has(.app-chip-mark) button:hover {
  border-color: var(--app-accent);
  color: var(--app-accent);
  background: var(--app-accent-soft);
}

.app-source-item {
  margin: 0 0 0.85rem 0;
  padding: 0 0 0.75rem 0;
  border-bottom: 1px solid var(--app-line);
}

.app-source-item:last-child {
  margin-bottom: 0.15rem;
  padding-bottom: 0;
  border-bottom: none;
}

.app-source-page {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--app-ink);
  letter-spacing: -0.01em;
  margin: 0 0 0.35rem 0;
}

.app-source-meta {
  font-weight: 400;
  font-size: 0.82rem;
  color: var(--app-muted);
}

[data-testid="stChatMessage"] {
  padding-top: 0.35rem;
  padding-bottom: 0.55rem;
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
  margin: 0 0 0.55rem 0;
  font-size: 0.92rem;
  line-height: 1.45;
  color: var(--app-slate);
}

[data-testid="stSidebar"] .app-sidebar-howto {
  margin: 0;
  font-size: 0.88rem;
  line-height: 1.45;
  color: var(--app-ink);
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
  font-size: 0.82rem;
  color: var(--app-muted);
}

[data-testid="stSidebar"] .app-sidebar-github {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--app-muted);
  text-decoration: none;
  padding: 0.35rem;
  border-radius: 0.35rem;
}

[data-testid="stSidebar"] .app-sidebar-github:hover {
  color: var(--app-ink);
  background: var(--app-accent-soft);
  text-decoration: none;
}

[data-testid="stSidebar"] .app-sidebar-github svg {
  display: block;
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


def inject_theme() -> None:
    """Inject demo theme CSS once per render (Streamlit-safe)."""
    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)
