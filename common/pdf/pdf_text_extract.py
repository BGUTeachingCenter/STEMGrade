from __future__ import annotations

from pathlib import Path


def _repair_common_mojibake(text: str) -> str:
    """
    Repair common mojibake from PDF text extraction.

    Example:
      ×¤×ª×¨×•×Ÿ  ->  פתרון
      âˆ’          ->  −
      â‰¤          ->  ≤

    Some PDFs store/expose text in a way that pypdf extracts as if UTF-8
    bytes were decoded through latin-1/cp1252.
    """
    if not text:
        return ""

    repaired = text

    # First try the common UTF-8-as-latin1 repair.
    # This fixes Hebrew patterns like ×¤×ª×¨×•×Ÿ.
    try:
        candidate = repaired.encode("latin1").decode("utf-8")
        if _looks_better_than(candidate, repaired):
            repaired = candidate
    except Exception:
        pass

    # Then fix common cp1252 math-symbol mojibake.
    # This fixes âˆ’, â‰¤, âˆˆ, etc.
    try:
        candidate = repaired.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        if _looks_better_than(candidate, repaired):
            repaired = candidate
    except Exception:
        pass

    return repaired


def _bad_mojibake_score(text: str) -> int:
    bad_fragments = [
        "×",
        "â",
        "Â",
        "Î",
        "Ï",
        "�",
    ]
    return sum(text.count(x) for x in bad_fragments)


def _hebrew_score(text: str) -> int:
    return sum(1 for ch in text if "\u0590" <= ch <= "\u05ff")


def _math_symbol_score(text: str) -> int:
    useful = "≤≥−∈∑∏∞"
    return sum(text.count(ch) for ch in useful)


def _looks_better_than(candidate: str, original: str) -> bool:
    if not candidate:
        return False

    original_bad = _bad_mojibake_score(original)
    candidate_bad = _bad_mojibake_score(candidate)

    original_good = _hebrew_score(original) + _math_symbol_score(original)
    candidate_good = _hebrew_score(candidate) + _math_symbol_score(candidate)

    return (
        candidate_bad < original_bad
        or candidate_good > original_good + 10
    )


def extract_pdf_text_layer(pdf_path: Path) -> str:
    """
    Extract embedded/selectable text from a non-handwritten PDF.

    Returns empty string if the PDF is scanned/image-only or extraction fails.
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _repair_common_mojibake(text).strip()

            if text:
                pages.append(f"% --- PDF page {i} ---\n{text}")

        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def looks_like_useful_pdf_text(text: str) -> bool:
    """
    Decide whether normal PDF extraction succeeded well enough.

    This is intentionally simple:
    - enough characters
    - contains question-like numbering or Hebrew solution placeholders
    """
    cleaned = _repair_common_mojibake(text or "").strip()

    if len(cleaned) < 200:
        return False

    has_question_numbers = any(
        marker in cleaned
        for marker in [".1", "1.", ".2", "2.", ".3", "3."]
    )

    has_hebrew_structure = "פתרון" in cleaned or "שאלה" in cleaned

    return has_question_numbers or has_hebrew_structure


def repair_pdf_text_mojibake(text: str) -> str:
    return _repair_common_mojibake(text or "")