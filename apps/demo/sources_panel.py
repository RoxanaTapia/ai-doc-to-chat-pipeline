"""Sources expander, answer cards, timing captions, and the developer checklist."""

from __future__ import annotations

import html
import re

import streamlit as st
from langchain_core.documents import Document

from ai_doc.rag.citations import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_sources_payload,
    is_insufficient_answer,
)
from page_copy import checked_all_pages_note
from page_preview import render_highlighted_page

_HEADING_RE = re.compile(r"^#{1,6}\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SKIP_HEADINGS = frozenset({"summary", "details"})
_EMPHASIS_RE = re.compile(r"(\*\*|__)(.*?)\1|([*_])(.*?)\3")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

EVAL_CHECKLIST_PREVIEW_CHARS = 80


def split_answer_headline(answer: str) -> tuple[str, str]:
    """First short line in large type; the rest stays as supporting text."""
    text = (answer or "").strip()
    if not text:
        return "", ""
    lines = [_HEADING_RE.sub("", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return text, ""
    if lines[0].lower() in _SKIP_HEADINGS and len(lines) > 1:
        lines = lines[1:]
    headline, remainder_lines = lines[0], lines[1:]
    remainder = "\n".join(remainder_lines).strip()
    if remainder_lines and remainder_lines[0].lower() in _SKIP_HEADINGS:
        remainder = "\n".join(remainder_lines[1:]).strip()
    if len(headline) > 80 and not remainder:
        parts = _SENTENCE_RE.split(text, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) <= 120:
            return parts[0].strip(), parts[1].strip()
    return headline, remainder


def strip_headline_markup(text: str) -> str:
    """Drop markdown emphasis so the big one-liner reads as plain type."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _EMPHASIS_RE.sub(lambda match: match.group(2) or match.group(4) or "", cleaned)
    return cleaned.strip()


def headline_number_in_quote(headline: str, quote: str) -> bool:
    """False when the one-liner names a number the quote does not contain."""
    numbers = _NUMBER_RE.findall(headline or "")
    if not numbers:
        return True
    haystack = quote or ""
    return all(number in haystack for number in numbers)


def display_answer_parts(answer: str, quote: str = "") -> tuple[str, str]:
    """Headline plus body, with markup stripped and a quote-number safety net."""
    headline, body = split_answer_headline(answer)
    headline = strip_headline_markup(headline)
    if quote and headline and not headline_number_in_quote(headline, quote):
        return "", strip_headline_markup(answer)
    return headline, body


def _build_sources_payload(
    retrieved_docs: list[Document],
    *,
    query: str | None = None,
    answer: str | None = None,
    display_max: int,
    preview_chars: int,
) -> list[dict]:
    """Pack page excerpts so the Sources expander can show them later."""
    return build_sources_payload(
        retrieved_docs,
        query=query,
        answer=answer,
        display_max=display_max,
        preview_chars=preview_chars,
        checklist_preview_chars=EVAL_CHECKLIST_PREVIEW_CHARS,
    )


def _question_preview(text: str, max_len: int = 52) -> str:
    """Shorten the question so a Sources title still fits."""
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1].rstrip() + "…"


def _dev_panel_title(base: str, question_preview: str | None, *, fallback: str) -> str:
    """Give each developer expander its own label, since Streamlit has no expander key."""
    preview = (question_preview or "").strip()
    if preview:
        return f"{base} — {_question_preview(preview, 40)}"
    return f"{base} — {fallback}"


def _sources_panel_title(message: dict) -> str:
    """Name the Sources expander after the question it answers."""
    preview = message.get("for_question")
    if preview:
        return f"Sources — {preview}"
    return "Sources — this answer"


def _paired_turns(messages: list[dict]) -> list[tuple[dict | None, dict]]:
    """Group chat history into (question, answer) pairs for answer cards."""
    pairs: list[tuple[dict | None, dict]] = []
    pending_user: dict | None = None
    for message in messages:
        if message.get("role") == "user":
            pending_user = message
            continue
        if message.get("role") == "assistant":
            pairs.append((pending_user, message))
            pending_user = None
    return pairs


def _toggle_page_preview(preview_id: str) -> None:
    """Open or close this page. on_click runs before any ask rerun can skip the button."""
    current = st.session_state.get("open_page_preview")
    st.session_state.open_page_preview = None if current == preview_id else preview_id


def _render_page_preview(pdf_bytes: bytes | None, source: dict, *, key: str) -> None:
    """Show the page with the passage highlighted when the quote is opened."""
    preview_id = f"{key}:{source.get('page')}:{source.get('preview', '')[:40]}"
    if st.button(
        "Show page",
        key=f"preview_{key}",
        help="Open this page with the passage highlighted.",
        on_click=_toggle_page_preview,
        args=(preview_id,),
    ):
        st.rerun()
    if st.session_state.get("open_page_preview") != preview_id:
        return
    png = render_highlighted_page(
        pdf_bytes or b"",
        source.get("page"),
        source.get("preview") or "",
    )
    if png:
        st.image(png, use_container_width=True)
    else:
        st.caption("Could not open that page from the current PDF.")


def _render_inline_sources(
    sources: list[dict],
    *,
    expanded: bool,
    pdf_bytes: bytes | None = None,
    card_key: str = "0",
) -> None:
    """Page badge and quote; Show page opens the highlighted page."""
    if not sources:
        return
    visible = sources if expanded else sources[:1]
    for source_idx, source in enumerate(visible):
        page = html.escape(str(source.get("page", "N/A")))
        preview = html.escape(source.get("preview") or "")
        st.markdown(
            f'<span class="app-page-badge">Page {page}</span>'
            f'<blockquote class="app-inline-quote">{preview}</blockquote>',
            unsafe_allow_html=True,
        )
        _render_page_preview(pdf_bytes, source, key=f"{card_key}_{source_idx}")
    extra = sources[len(visible) :]
    if extra and not expanded:
        with st.expander("More pages", expanded=False):
            for source_idx, source in enumerate(extra, start=len(visible)):
                page = html.escape(str(source.get("page", "N/A")))
                preview = html.escape(source.get("preview") or "")
                st.markdown(
                    f'<span class="app-page-badge">Page {page}</span>'
                    f'<blockquote class="app-inline-quote">{preview}</blockquote>',
                    unsafe_allow_html=True,
                )
                _render_page_preview(pdf_bytes, source, key=f"{card_key}_{source_idx}")


def _render_answer_card(
    *,
    question: str,
    answer: str,
    sources: list[dict] | None,
    refusal: bool | None = None,
    expanded: bool = False,
    timing: dict[str, float] | None = None,
    developer_mode: bool = False,
    page_count: int = 0,
    any_blank_pages: bool = False,
    pdf_bytes: bytes | None = None,
    card_key: str = "0",
) -> None:
    """Muted question, one-line headline, page badge, or a calm refusal."""
    is_refusal = bool(refusal) or is_insufficient_answer(answer, question)
    marker = "app-answer-card--refusal" if is_refusal else "app-answer-card"
    card_kind = "refusal_card" if is_refusal else "answer_card"
    with st.container(key=f"{card_kind}_{card_key}"):
        st.markdown(
            f'<p class="app-answer-q {marker}">{html.escape(question)}</p>',
            unsafe_allow_html=True,
        )
        if is_refusal:
            st.markdown(
                f'<p class="app-answer-headline">{html.escape(INSUFFICIENT_CONTEXT_ANSWER)}</p>',
                unsafe_allow_html=True,
            )
            note = checked_all_pages_note(page_count, any_blank=any_blank_pages)
            st.markdown(
                f'<p class="app-no-sources">{html.escape(note)}</p>',
                unsafe_allow_html=True,
            )
        else:
            quote = (sources[0].get("preview") if sources else "") or ""
            headline, body = display_answer_parts(answer, quote)
            # Escape body + timing in one markdown call so the timing line cannot
            # paint a frame after the body. Markup from the model/PDF cannot inject.
            timing_line = _answer_timing_line(timing, developer_mode=developer_mode)
            if headline:
                st.markdown(
                    f'<p class="app-answer-headline">{html.escape(headline)}</p>',
                    unsafe_allow_html=True,
                )
            body_parts: list[str] = []
            if body:
                body_parts.append(html.escape(body))
            if timing_line:
                body_parts.append(
                    f'<p class="app-timing">{html.escape(timing_line)}</p>'
                )
            if body_parts:
                st.markdown("\n\n".join(body_parts), unsafe_allow_html=True)
            if sources:
                _render_inline_sources(
                    sources,
                    expanded=expanded,
                    pdf_bytes=pdf_bytes,
                    card_key=card_key,
                )


def _render_sources_panel(
    sources: list[dict],
    *,
    developer_mode: bool,
    title: str = "Sources",
    expanded: bool = False,
) -> None:
    """Show the page and excerpt so the visitor can check the answer."""
    if not sources:
        return
    with st.expander(title, expanded=expanded):
        st.caption("Page and short excerpt from the document — verify against the answer.")
        blocks: list[str] = []
        for source in sources:
            page = source.get("page", "N/A")
            preview = html.escape(source.get("preview") or "")
            meta = ""
            if developer_mode and source.get("score") != "N/A":
                meta = (
                    f' <span class="app-source-meta">· relevance '
                    f"{html.escape(str(source['score']))}</span>"
                )
            blocks.append(
                '<div class="app-source-item">'
                f'<div class="app-source-page">Page {html.escape(str(page))}{meta}</div>'
                f'<blockquote class="app-source-quote">{preview}</blockquote>'
                "</div>"
            )
        st.markdown("".join(blocks), unsafe_allow_html=True)


def _on_section_label(on_section: bool | None) -> str:
    """Turn the on-section flag into the checklist word."""
    if on_section is True:
        return "Yes"
    if on_section is False:
        return "No"
    return "N/A"


def _render_source_checklist(
    sources: list[dict],
    *,
    target_section: str | None = None,
    title: str = "Source checklist (eval)",
) -> None:
    """Show page, excerpt, and on-section so a developer can score the answer."""
    if not sources:
        return
    with st.expander(title, expanded=False):
        if target_section:
            st.caption(
                f"Target section: **{target_section}** · auto-tag is heuristic — "
                "confirm manually when scoring."
            )
        else:
            st.caption("No section in question — on-section column shows N/A.")
        for source_idx, source in enumerate(sources, start=1):
            preview = source.get("checklist_preview") or source.get("preview", "")
            st.markdown(
                f"{source_idx}. **Page {source['page']}** · "
                f"on-section {_on_section_label(source.get('on_section'))}  \n"
                f"> {preview}"
            )
        if target_section:
            on_count = sum(1 for source in sources if source.get("on_section") is True)
            st.caption(
                f"Auto on-section count: **{on_count}/{len(sources)}** "
                "(Round 4 bar: ≥3/5 on-section for Q1 and Q2)."
            )


def _render_eval_context_panel(
    eval_context: str,
    *,
    target_section: str | None = None,
    chunk_count: int | None = None,
    title: str = "Exact context fed to LLM",
    top_k: int,
) -> None:
    """Keep the exact context visible so a developer can see what the model was given."""
    with st.expander(title, expanded=False):
        st.code(eval_context, language="text")
        chunk_note = f" • {chunk_count} chunks" if chunk_count else ""
        st.caption(f"• {len(eval_context)} chars{chunk_note} · top-k={top_k}")
        if target_section == "3":
            st.info(
                "Round 4 Q2 diagnostic: if return/destroy or termination language "
                "appears **in this context**, retrieval is likely at fault; "
                "if Section 3 duties are here but missing from the answer, "
                "generation is likely at fault."
            )


def _format_elapsed_ms(elapsed_ms: float) -> str:
    """Turn milliseconds into a short caption."""
    seconds = max(0.0, elapsed_ms / 1000.0)
    if seconds < 60:
        # Sub-0.1s refusals (context guard, no model call) must not read as 0.0s.
        if seconds < 0.1:
            return f"{seconds:.2f}s"
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(round(seconds % 60))
    if rem == 60:
        minutes += 1
        rem = 0
    return f"{minutes}m {rem}s" if rem else f"{minutes}m"


def _response_timing_caption(
    *,
    total_ms: float,
    retrieval_ms: float = 0.0,
    generation_ms: float = 0.0,
    developer_mode: bool = False,
) -> str:
    """Say how long the answer took, with a split only in developer mode."""
    total_label = _format_elapsed_ms(total_ms)
    if not developer_mode:
        return f"Answered in {total_label}"
    parts = []
    if retrieval_ms > 0:
        parts.append(f"retrieval {_format_elapsed_ms(retrieval_ms)}")
    if generation_ms > 0:
        parts.append(f"generation {_format_elapsed_ms(generation_ms)}")
    if parts:
        return f"Answered in {total_label} ({' · '.join(parts)})"
    return f"Answered in {total_label}"


def _remember_response_timing(timing: dict[str, float] | None) -> None:
    """Store the last timing so the caption can show it after a rerun."""
    if timing and timing.get("total_ms", 0) > 0:
        st.session_state.last_response_timing = timing


def _answer_timing_line(timing: dict[str, float] | None, *, developer_mode: bool) -> str:
    """Plain timing caption, or empty when there is nothing to show."""
    if not timing or timing.get("total_ms", 0) <= 0:
        return ""
    caption = _response_timing_caption(
        total_ms=timing["total_ms"],
        retrieval_ms=timing.get("retrieval_ms", 0.0),
        generation_ms=timing.get("generation_ms", 0.0),
        developer_mode=developer_mode,
    )
    # Hide a true zero that would still format as 0.00s.
    if caption.startswith("Answered in 0.00s"):
        return ""
    return caption


def _answer_timing_html(timing: dict[str, float] | None, *, developer_mode: bool) -> str:
    """HTML for the timing caption, or empty when there is nothing to show."""
    caption = _answer_timing_line(timing, developer_mode=developer_mode)
    if not caption:
        return ""
    return f'<p class="app-timing">{html.escape(caption)}</p>'


def _render_answer_timing(timing: dict[str, float] | None, *, developer_mode: bool) -> None:
    """Put the timing caption under the answer, not inside it."""
    caption = _answer_timing_line(timing, developer_mode=developer_mode)
    if caption:
        st.markdown(
            f'<p class="app-timing">{html.escape(caption)}</p>',
            unsafe_allow_html=True,
        )
