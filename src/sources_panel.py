"""Sources expander, timing captions, and the developer checklist."""

from __future__ import annotations

import html

import streamlit as st
from langchain_core.documents import Document

from rag.citations import build_sources_payload

EVAL_CHECKLIST_PREVIEW_CHARS = 80


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


def _render_answer_timing(timing: dict[str, float] | None, *, developer_mode: bool) -> None:
    """Put the timing caption under the answer, not inside it."""
    if not timing or timing.get("total_ms", 0) <= 0:
        return
    st.caption(
        _response_timing_caption(
            total_ms=timing["total_ms"],
            retrieval_ms=timing.get("retrieval_ms", 0.0),
            generation_ms=timing.get("generation_ms", 0.0),
            developer_mode=developer_mode,
        )
    )
