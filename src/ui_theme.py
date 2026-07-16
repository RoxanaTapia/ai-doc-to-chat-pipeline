"""Minimal Streamlit-safe theme tokens for client-facing demos.

Calm neutrals (ink / slate / soft paper). ``--app-accent`` is a future hook —
leave unused until a brand accent is chosen; do not invent a logo here.
"""

from __future__ import annotations

import streamlit as st

# Cool neutrals — restrained, buyer-safe; not purple-gradient / glow chrome.
_THEME_CSS = """
:root {
  --app-ink: #1c2430;
  --app-slate: #5c6570;
  --app-muted: #8a929c;
  --app-paper: #f5f6f8;
  --app-accent: #3d5a6c;
}

section.main h1 {
  color: var(--app-ink);
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.25;
  margin-bottom: 0.2rem;
}

.app-hero {
  margin: 0 0 1.35rem 0;
  padding: 0 0 1rem 0;
  border-bottom: 1px solid color-mix(in srgb, var(--app-ink) 10%, var(--app-paper));
}

.app-hero__lead {
  font-size: 1.08rem;
  line-height: 1.55;
  color: var(--app-slate);
  margin: 0.15rem 0 0.5rem 0;
}

.app-hero__kicker {
  font-size: 0.9rem;
  line-height: 1.45;
  color: var(--app-muted);
  margin: 0;
  letter-spacing: 0.01em;
}
"""


def inject_theme() -> None:
    """Inject calm foundation CSS once per render (Streamlit-safe)."""
    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)
