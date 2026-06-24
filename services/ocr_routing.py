from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz

DocumentMode = Literal["auto", "printed", "handwritten"]
DocumentType = Literal["native_text_pdf", "printed_scan", "handwritten_scan", "image", "unknown"]


@dataclass(frozen=True)
class OcrRouteDecision:
    requested_mode: str
    document_type: DocumentType
    ocr_path: str
    reason: str
    native_text: str = ""
    native_text_char_count: int = 0
    text_layer_pages: int = 0


def _extract_pdf_text(path: Path, *, max_pages: int | None = None) -> tuple[str, int]:
    doc = fitz.open(str(path))
    try:
        chunks: list[str] = []
        page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(page_count):
            txt = doc[i].get_text("text") or ""
            if txt.strip():
                chunks.append(txt.strip())
        return "\n\n".join(chunks).strip(), page_count
    finally:
        doc.close()


def decide_student_ocr_path(path: Path, requested_mode: str = "auto") -> OcrRouteDecision:
    mode = (requested_mode or "auto").strip().lower()
    if mode not in {"auto", "printed", "handwritten"}:
        mode = "auto"

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        sample_text, sample_pages = _extract_pdf_text(path, max_pages=3)
        sample_chars = len(sample_text.strip())

        if mode == "printed":
            if sample_chars >= 200:
                full_text, _ = _extract_pdf_text(path, max_pages=None)
                return OcrRouteDecision(
                    requested_mode=mode,
                    document_type="native_text_pdf",
                    ocr_path="native_pdf_text_layer",
                    reason="Manual printed mode and the PDF contains a usable text layer.",
                    native_text=full_text,
                    native_text_char_count=len(full_text),
                    text_layer_pages=sample_pages,
                )
            return OcrRouteDecision(
                requested_mode=mode,
                document_type="printed_scan",
                ocr_path="printed_scan_ocr",
                reason="Manual printed mode but the PDF text layer is missing or too small.",
                native_text_char_count=sample_chars,
                text_layer_pages=sample_pages,
            )

        if mode == "handwritten":
            return OcrRouteDecision(
                requested_mode=mode,
                document_type="handwritten_scan",
                ocr_path="handwritten_scan_ocr",
                reason="Manual handwritten mode selected.",
                native_text_char_count=sample_chars,
                text_layer_pages=sample_pages,
            )

        # Auto mode.
        if sample_chars >= 500:
            full_text, _ = _extract_pdf_text(path, max_pages=None)
            return OcrRouteDecision(
                requested_mode=mode,
                document_type="native_text_pdf",
                ocr_path="native_pdf_text_layer",
                reason="Auto detected a usable embedded PDF text layer; OCR is skipped.",
                native_text=full_text,
                native_text_char_count=len(full_text),
                text_layer_pages=sample_pages,
            )

        return OcrRouteDecision(
            requested_mode=mode,
            document_type="handwritten_scan",
            ocr_path="handwritten_scan_ocr",
            reason="Auto detected an image-only or low-text PDF; using handwritten scan path.",
            native_text_char_count=sample_chars,
            text_layer_pages=sample_pages,
        )

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if mode == "printed":
            return OcrRouteDecision(
                requested_mode=mode,
                document_type="printed_scan",
                ocr_path="printed_scan_ocr",
                reason="Manual printed mode selected for image upload.",
            )
        return OcrRouteDecision(
            requested_mode=mode,
            document_type="handwritten_scan",
            ocr_path="handwritten_scan_ocr",
            reason="Image upload; using handwritten scan path unless printed override is selected.",
        )

    return OcrRouteDecision(
        requested_mode=mode,
        document_type="unknown",
        ocr_path="generic_ocr",
        reason=f"No specific OCR route for suffix {suffix!r}; using generic OCR.",
    )


def decide_bank_ocr_path(path: Path, requested_mode: str = "printed") -> OcrRouteDecision:
    """Alias for teacher solution-bank uploads.

    The routing logic is shared with student OCR: printed mode can use a native
    PDF text layer, handwritten mode forces vision/OCR over page images, and
    auto mode chooses based on the PDF text layer.
    """
    return decide_student_ocr_path(path, requested_mode=requested_mode)
