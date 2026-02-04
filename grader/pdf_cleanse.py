# grader/pdf_cleanse.py
from __future__ import annotations

"""Clean a reference/test PDF before grading/bundling.

Most university exam PDFs contain extra pages you usually don't want to include
inside a Q/A bundle:

* Opening instructions / cover sheet
* A formula sheet (דף נוסחאות)
* Intentionally blank pages at the end

This module provides `cleanse_test_pdf()` which produces a cleaned PDF and a
small debug report explaining what was removed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import fitz  # PyMuPDF


# -----------------------------
# Detection heuristics
# -----------------------------

_HEB_INTENTIONAL_BLANK = "דף ריק"
_HEB_FORMULA_SHEET = "דף נוסחאות"


def _extract_first_question_like(text: str) -> Optional[int]:
    """Best-effort: detect the first question number on a page.

    We intentionally keep this local (instead of importing qa_bundle)
    so this module can be used independently.
    """
    import re

    # Common styles we see in these PDFs:
    #  1) ".1 (35 נקודות)"  or  "1. (35 נקודות)"
    #  2) "שאלה 1" or "Question 1"
    m = re.search(r"\(נקודות\s*\d+\)\s*\.?\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:שאלה|Question)\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Sometimes the PDF starts the question with just ".1" near the top.
    m = re.search(r"(?:^|\n)\s*\.?\s*(\d{1,2})\s*(?:\(|\.)", text)
    if m:
        return int(m.group(1))
    return None


def _looks_like_formula_sheet(text: str) -> bool:
    t = (text or "").replace("\u200f", "").replace("\u200e", "")
    # Strong indicators (headings that typically only appear on the sheet)
    if ("נוסחאות טריגונומטריות" in t) or ("אינטגרלים מיידיים" in t):
        return True
    if ("Formula" in t) and ("sheet" in t.lower()):
        return True

    # "דף נוסחאות" can appear as a *reference* inside solutions (e.g., "ראו דף נוסחאות").
    # Treat it as a formula sheet only if it looks like a page heading and not a mention.
    if _HEB_FORMULA_SHEET in t:
        head = t.strip()[:120]
        if head.startswith(_HEB_FORMULA_SHEET):
            return True
        # If the page doesn't look like a question page, it's likely the sheet.
        if _extract_first_question_like(t) is None:
            # still require "דף נוסחאות" to be near the top
            if _HEB_FORMULA_SHEET in t[:400]:
                return True
    return False


def _looks_like_intentional_blank(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return _HEB_INTENTIONAL_BLANK in t


def _page_has_visible_marks(page: fitz.Page, *, sample_dpi: int = 72, nonwhite_threshold: int = 250,
                            min_nonwhite_fraction: float = 0.002) -> bool:
    """Image-based check for pages that have no selectable text.

    Returns True if the rasterized page seems to contain visible ink.
    """
    # Low DPI is enough to detect any meaningful content quickly.
    mat = fitz.Matrix(sample_dpi / 72.0, sample_dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    samples = pix.samples  # bytes
    # Count pixels that aren't near-white.
    # Each pixel is RGB = 3 bytes.
    total_px = pix.width * pix.height
    if total_px == 0:
        return False

    nonwhite = 0
    # Iterate in steps to keep it fast without numpy.
    step = 12  # sample every 4th pixel (4*3 bytes)
    for i in range(0, len(samples), step):
        r = samples[i]
        g = samples[i + 1]
        b = samples[i + 2]
        if r < nonwhite_threshold or g < nonwhite_threshold or b < nonwhite_threshold:
            nonwhite += 1
    sampled_px = len(samples) // step
    return (nonwhite / max(sampled_px, 1)) >= min_nonwhite_fraction


def _is_blank_page(page: fitz.Page) -> bool:
    """Decide if a page is blank (or intentionally blank)."""
    text = page.get_text("text") or ""
    if _looks_like_intentional_blank(text):
        # If it has *some* text but it's the intentional blank marker, treat as blank.
        return True
    if text.strip():
        return False
    # No text. Use raster check (helps for scanned PDFs).
    return not _page_has_visible_marks(page)


# -----------------------------
# Public API
# -----------------------------


@dataclass(frozen=True)
class CleanseReport:
    input_pdf: Path
    output_pdf: Path
    removed_pages_1based: Tuple[int, ...]
    kept_pages_1based: Tuple[int, ...]
    reason_lines: Tuple[str, ...]
    debug_report_path: Optional[Path] = None


def cleanse_test_pdf(
    input_pdf: Path,
    out_dir: Path,
    *,
    output_name: Optional[str] = None,
    remove_instructions: bool = True,
    remove_formula_sheet: bool = True,
    remove_trailing_blank_pages: bool = True,
    extra_remove_text_markers: Sequence[str] = (),
    write_debug_report: bool = True,
) -> CleanseReport:
    """Create a cleaned copy of `input_pdf`.

    Heuristics:
      1) Instruction pages: drop everything before the first page where a question number is detected.
      2) Formula sheet: drop pages containing "דף נוסחאות" (and a couple of backups).
      3) Trailing blanks: drop blank pages from the end.

    Returns a `CleanseReport` with kept/removed pages (1-based).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_pdf = Path(input_pdf)

    if output_name is None:
        output_name = f"{input_pdf.stem}_clean.pdf"
    out_pdf = out_dir / output_name
    dbg_path = out_dir / f"{Path(output_name).stem}_cleanse_report.txt"

    doc = fitz.open(str(input_pdf))
    n = doc.page_count
    keep = [True] * n
    reasons: List[str] = []

    # 1) Instructions / cover pages
    if remove_instructions:
        first_q_page: Optional[int] = None
        for i in range(n):
            t = doc[i].get_text("text") or ""
            q = _extract_first_question_like(t)
            if q is not None:
                first_q_page = i
                break
        if first_q_page is not None and first_q_page > 0:
            for i in range(0, first_q_page):
                keep[i] = False
            reasons.append(
                f"Removed instruction/cover pages: 1..{first_q_page} (first detected question page is {first_q_page + 1})."
            )
        elif first_q_page is None:
            reasons.append(
                "Could not detect a first question page via text heuristics; instruction-page removal skipped."
            )

    # 2) Formula sheet
    if remove_formula_sheet:
        removed_formula: List[int] = []
        for i in range(n):
            if not keep[i]:
                continue
            t = doc[i].get_text("text") or ""
            if _looks_like_formula_sheet(t):
                keep[i] = False
                removed_formula.append(i)
        if removed_formula:
            pages = ", ".join(str(i + 1) for i in removed_formula)
            reasons.append(f"Removed formula-sheet pages (matched '{_HEB_FORMULA_SHEET}' heuristics): {pages}.")

    # 3) Extra text markers removal
    if extra_remove_text_markers:
        removed_extra: List[int] = []
        for i in range(n):
            if not keep[i]:
                continue
            t = doc[i].get_text("text") or ""
            if any(m in t for m in extra_remove_text_markers):
                keep[i] = False
                removed_extra.append(i)
        if removed_extra:
            pages = ", ".join(str(i + 1) for i in removed_extra)
            reasons.append(f"Removed pages by extra text markers {list(extra_remove_text_markers)}: {pages}.")

    # 4) Trailing blank pages
    if remove_trailing_blank_pages:
        removed_trailing: List[int] = []
        i = n - 1
        while i >= 0:
            if keep[i] and _is_blank_page(doc[i]):
                keep[i] = False
                removed_trailing.append(i)
                i -= 1
                continue
            # stop once we hit a non-blank kept page
            if keep[i]:
                break
            i -= 1
        if removed_trailing:
            removed_trailing_sorted = sorted([x + 1 for x in removed_trailing])
            reasons.append(f"Removed trailing blank pages: {removed_trailing_sorted}.")

    kept_pages = [i for i, k in enumerate(keep) if k]
    removed_pages = [i for i, k in enumerate(keep) if not k]

    if not kept_pages:
        raise RuntimeError(
            "PDF cleansing removed all pages. "
            "Disable one of the removal options or adjust heuristics/text markers."
        )

    # Write cleaned PDF
    new_doc = fitz.open()
    for i in kept_pages:
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
    new_doc.save(str(out_pdf))
    new_doc.close()
    doc.close()

    # Debug report
    if write_debug_report:
        lines: List[str] = []
        lines.append(f"Input:  {input_pdf}")
        lines.append(f"Output: {out_pdf}")
        lines.append("")
        lines.extend(reasons or ["No pages were removed (all heuristics kept everything)."])
        lines.append("")
        lines.append(f"Kept pages (1-based): {[i+1 for i in kept_pages]}")
        lines.append(f"Removed pages (1-based): {[i+1 for i in removed_pages]}")
        dbg_path.write_text("\n".join(lines), encoding="utf-8")
        dbg_report = dbg_path
    else:
        dbg_report = None

    return CleanseReport(
        input_pdf=input_pdf,
        output_pdf=out_pdf,
        removed_pages_1based=tuple(i + 1 for i in removed_pages),
        kept_pages_1based=tuple(i + 1 for i in kept_pages),
        reason_lines=tuple(reasons),
        debug_report_path=dbg_report,
    )
