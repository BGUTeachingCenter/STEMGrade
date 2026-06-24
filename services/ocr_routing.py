# services/ocr_routing.py
"""
Decide which OCR path to use for an uploaded file.

Two entry points share one decision routine:
  * decide_student_ocr_path  - student submissions (default mode "auto")
  * decide_bank_ocr_path     - teacher solution-bank uploads (printed/handwritten)

The decision distinguishes:
  * native_pdf_text_layer  - PDF already has selectable text; no AI OCR needed
  * handwritten_scan_ocr   - image-only / low-text scan, handwritten path
  * generic_ocr            - everything else (printed scan, images, fallback)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# A PDF with at least this many extractable characters is treated as having a
# usable native text layer (printed/typed document). Kept low so a mostly-image
# PDF with a few stray characters still routes to OCR.
MIN_NATIVE_TEXT_CHARS = 40

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass
class OcrRouteDecision:
    requested_mode: str
    document_type: str          # "printed" | "handwritten_scan" | "image"
    ocr_path: str               # "native_pdf_text_layer" | "handwritten_scan_ocr" | "generic_ocr"
    reason: str
    native_text: str = ""
    native_text_char_count: int = 0
    text_layer_pages: int = 0


def _extract_pdf_native_text(path: Path) -> tuple[str, int]:
    """Return (full_text, pages_with_text). Empty/0 if not a readable PDF."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return "", 0

    try:
        doc = fitz.open(str(path))
    except Exception:
        return "", 0

    try:
        chunks: list[str] = []
        pages_with_text = 0
        for page in doc:
            txt = (page.get_text("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if txt:
                pages_with_text += 1
                chunks.append(txt)
        return "\n\n".join(chunks).strip(), pages_with_text
    finally:
        doc.close()


def _normalize_mode(requested_mode: str | None) -> str:
    v = (requested_mode or "auto").strip().lower()
    handwritten = {"hand", "handwrite", "handwritten", "handwritten_scan", "scan"}
    printed = {"print", "printed", "typed", "typed_pdf", "native", "printed_scan"}
    if v in handwritten:
        return "handwritten"
    if v in printed:
        return "printed"
    return "auto"


def _decide(path: Path, requested_mode: str | None) -> OcrRouteDecision:
    requested = (requested_mode or "auto").strip() or "auto"
    mode = _normalize_mode(requested)
    suffix = path.suffix.lower()

    # Images never carry a text layer.
    if suffix in _IMAGE_SUFFIXES:
        if mode == "printed":
            return OcrRouteDecision(
                requested_mode=requested,
                document_type="image",
                ocr_path="generic_ocr",
                reason="Image upload marked printed; using generic OCR.",
            )
        return OcrRouteDecision(
            requested_mode=requested,
            document_type="handwritten_scan",
            ocr_path="handwritten_scan_ocr",
            reason="Image upload; using handwritten scan OCR path.",
        )

    native_text, text_layer_pages = ("", 0)
    char_count = 0
    if suffix == ".pdf":
        native_text, text_layer_pages = _extract_pdf_native_text(path)
        char_count = len(native_text)

    has_text_layer = suffix == ".pdf" and char_count >= MIN_NATIVE_TEXT_CHARS

    # Explicit handwritten request always goes to the handwritten path.
    if mode == "handwritten":
        return OcrRouteDecision(
            requested_mode=requested,
            document_type="handwritten_scan",
            ocr_path="handwritten_scan_ocr",
            reason="Requested handwritten mode; using handwritten scan OCR path.",
            native_text=native_text,
            native_text_char_count=char_count,
            text_layer_pages=text_layer_pages,
        )

    # Printed or auto: prefer the native text layer when present.
    if has_text_layer:
        return OcrRouteDecision(
            requested_mode=requested,
            document_type="printed",
            ocr_path="native_pdf_text_layer",
            reason="PDF has a usable native text layer; skipping AI OCR.",
            native_text=native_text,
            native_text_char_count=char_count,
            text_layer_pages=text_layer_pages,
        )

    if mode == "printed":
        return OcrRouteDecision(
            requested_mode=requested,
            document_type="printed",
            ocr_path="generic_ocr",
            reason="Printed document without a usable text layer; using generic OCR.",
            native_text=native_text,
            native_text_char_count=char_count,
            text_layer_pages=text_layer_pages,
        )

    # auto + no usable text layer -> assume image-only / handwritten scan.
    return OcrRouteDecision(
        requested_mode=requested,
        document_type="handwritten_scan",
        ocr_path="handwritten_scan_ocr",
        reason="Auto detected an image-only or low-text PDF; using handwritten scan path.",
        native_text=native_text,
        native_text_char_count=char_count,
        text_layer_pages=text_layer_pages,
    )


def decide_student_ocr_path(path: Path | str, requested_mode: str | None = "auto") -> OcrRouteDecision:
    """Route a student submission. Default mode is auto-detection."""
    return _decide(Path(path), requested_mode)


def decide_bank_ocr_path(path: Path | str, requested_mode: str | None = "printed") -> OcrRouteDecision:
    """Route a teacher solution-bank upload (printed/handwritten)."""
    return _decide(Path(path), requested_mode)
