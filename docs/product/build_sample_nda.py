"""Rebuild docs/product/sample-nda.pdf with a filmable layout.

Keeps term on page 2, governing law on page 3, and Remedies with its body.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUT = Path(__file__).with_name("sample-nda.pdf")
PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X = 64.0
MARGIN_TOP = 62.0
MARGIN_BOTTOM = 56.0
FOOTER_Y = 36.0
BODY_SIZE = 11.0
BODY_LEADING = 15.0
HEAD_SIZE = 12.0
TITLE_SIZE = 13.0


def _width() -> float:
    return PAGE_W - 2 * MARGIN_X


def _new_page(doc: pymupdf.Document) -> pymupdf.Page:
    return doc.new_page(width=PAGE_W, height=PAGE_H)


def _footer(page: pymupdf.Page, number: int) -> None:
    page.insert_text(
        (MARGIN_X, PAGE_H - FOOTER_Y),
        "Sample confidentiality agreement (fictional - for demo use only)",
        fontname="helv",
        fontsize=8,
        color=(0.36, 0.40, 0.44),
    )
    page.insert_text(
        (PAGE_W - MARGIN_X - 8, PAGE_H - FOOTER_Y),
        str(number),
        fontname="helv",
        fontsize=8,
        color=(0.36, 0.40, 0.44),
    )


def _draw_text(
    page: pymupdf.Page,
    y: float,
    text: str,
    *,
    fontname: str = "times-roman",
    fontsize: float = BODY_SIZE,
    leading: float = BODY_LEADING,
    color: tuple[float, float, float] = (0.10, 0.12, 0.14),
) -> float:
    rect = pymupdf.Rect(MARGIN_X, y, PAGE_W - MARGIN_X, PAGE_H - MARGIN_BOTTOM)
    used = page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontsize=fontsize,
        align=pymupdf.TEXT_ALIGN_LEFT,
    )
    # insert_textbox returns unused height; consumed = rect height - unused.
    consumed = rect.height - used
    # Keep a stable step even when the last line is short.
    return y + max(consumed, leading if text else 0.0)


def _heading(page: pymupdf.Page, y: float, text: str) -> float:
    y = _draw_text(
        page,
        y,
        text,
        fontname="helv",
        fontsize=HEAD_SIZE,
        leading=HEAD_SIZE + 4,
        color=(0.10, 0.12, 0.14),
    )
    return y + 4


def _body(page: pymupdf.Page, y: float, text: str) -> float:
    return _draw_text(page, y + 2, text) + 8


def _bullet(page: pymupdf.Page, y: float, text: str) -> float:
    return _draw_text(page, y, f"-  {text}") + 2


def build() -> Path:
    doc = pymupdf.open()

    # --- Page 1: title, parties, sections 1–2 ---
    page = _new_page(doc)
    y = MARGIN_TOP
    title = "CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT"
    title_w = pymupdf.get_text_length(title, fontname="helv", fontsize=TITLE_SIZE)
    if title_w > _width():
        raise RuntimeError("Title no longer fits on one line; reduce TITLE_SIZE.")
    page.insert_text(
        (MARGIN_X, y + TITLE_SIZE),
        title,
        fontname="helv",
        fontsize=TITLE_SIZE,
        color=(0.10, 0.12, 0.14),
    )
    y += TITLE_SIZE + 16
    y = _draw_text(
        page,
        y,
        "Sample confidentiality agreement (fictional - for demo use only).",
        fontname="helv",
        fontsize=9.5,
        leading=13,
        color=(0.36, 0.40, 0.44),
    )
    y += 10
    y = _body(
        page,
        y,
        "This document is a fictional non-disclosure agreement included solely "
        "to let you test the pilot at ai-doc-pilot.roxanatapia.dev. All parties, "
        "amounts, and dates are invented. Do not rely on it for legal purposes.",
    )
    y = _body(
        page,
        y,
        "This Agreement is entered into as of January 1, 2025, between Acme Corp "
        '("Disclosing Party") and Beta Labs Ltd ("Receiving Party").',
    )
    y = _heading(page, y + 6, "Section 1 - Definition of Confidential Information")
    y = _body(
        page,
        y,
        '"Confidential Information" means any information disclosed by the '
        "Disclosing Party that has commercial value and is labeled "
        '"Confidential", including but not limited to: business plans, '
        "technical data, customer lists, financial projections, product "
        "roadmaps, pricing strategies, and marketing materials. Information is "
        "deemed confidential if marked as such or understood by a reasonable "
        "person to be so.",
    )
    y = _heading(page, y + 4, "Section 2 - Exclusions")
    y = _body(page, y, "The obligations of this Agreement do not apply to information that:")
    y = _bullet(
        page, y, "is or becomes publicly available through no fault of the Receiving Party;"
    )
    y = _bullet(page, y, "was rightfully known to the Receiving Party before disclosure;")
    y = _bullet(
        page,
        y,
        "is independently developed by the Receiving Party without use of "
        "Confidential Information;",
    )
    y = _bullet(
        page,
        y,
        "is required to be disclosed by law or court order, provided prompt "
        "written notice is given.",
    )
    _footer(page, 1)

    # --- Page 2: obligations, term, oral disclosures ---
    page = _new_page(doc)
    y = MARGIN_TOP
    y = _heading(page, y, "Section 3 - Obligations of the Receiving Party")
    y = _body(page, y, "The Receiving Party shall:")
    y = _bullet(page, y, "Hold all Confidential Information in strict confidence;")
    y = _bullet(
        page,
        y,
        "Restrict access to only those employees, contractors, and third "
        "parties who have a legitimate need to know and who have signed "
        "identical nondisclosure restrictions;",
    )
    y = _bullet(
        page,
        y,
        "Not use the information for any purpose other than the evaluation of a "
        "potential business relationship;",
    )
    y = _bullet(
        page,
        y,
        "Return or destroy all materials containing Confidential Information "
        "upon written request of the Disclosing Party.",
    )
    y = _heading(page, y + 10, "Section 4 - Term")
    y = _body(
        page,
        y,
        "This Agreement shall remain in effect for a period of three (3) years "
        "from the date of signing. The obligations of confidentiality shall "
        "survive termination.",
    )
    y = _heading(page, y + 8, "Section 5 - Oral Disclosures")
    y = _body(
        page,
        y,
        "Any oral disclosure of Confidential Information must be confirmed in "
        "writing by the Disclosing Party within ten (10) business days of the "
        "disclosure. The Receiving Party further covenants that it will instruct "
        "all personnel who receive oral disclosures to treat such information "
        "with the highest degree of care and to refrain from discussing it in "
        "any public or semi-public setting.",
    )
    _footer(page, 2)

    # --- Page 3: remedies (heading + body together), governing law, signatures ---
    page = _new_page(doc)
    y = MARGIN_TOP
    y = _heading(page, y, "Section 6 - Remedies")
    y = _body(
        page,
        y,
        "The Receiving Party acknowledges that any breach of this Agreement "
        "would cause irreparable harm to the Disclosing Party's market position "
        "and intellectual property rights. Any attempt by the Receiving Party to "
        "assert a shorter duration or to argue that certain information has lost "
        "its confidential character must be supported by clear and convincing "
        "evidence. The Disclosing Party is entitled to seek injunctive relief in "
        "addition to any other remedies available at law or equity.",
    )
    y = _heading(page, y + 8, "Section 7 - Governing Law")
    y = _body(
        page,
        y,
        "This Agreement shall be governed by the laws of Spain. Any dispute "
        "arising hereunder shall be resolved exclusively in the courts of Madrid.",
    )
    y = _heading(page, y + 8, "Section 8 - Entire Agreement")
    y = _body(
        page,
        y,
        "This Agreement constitutes the entire understanding between the parties "
        "with respect to its subject matter and supersedes all prior "
        "negotiations, representations, and agreements.",
    )
    y += 16
    y = _draw_text(
        page,
        y,
        "Signed on behalf of Acme Corp: ___________________    Date: __________",
        fontname="times-italic",
        fontsize=11,
    )
    y += 8
    _draw_text(
        page,
        y,
        "Signed on behalf of Beta Labs Ltd: ___________________    Date: __________",
        fontname="times-italic",
        fontsize=11,
    )
    _footer(page, 3)

    OUT.write_bytes(doc.tobytes())
    doc.close()
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
