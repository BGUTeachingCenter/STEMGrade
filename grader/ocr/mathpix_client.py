from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

MATHPIX_TEXT_URL = "https://api.mathpix.com/v3/text"


class MathpixError(RuntimeError):
    pass


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _to_data_url(path: Path) -> str:
    mime = _guess_mime(path)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def process_image_or_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    include_line_data: bool = True,
) -> dict[str, Any]:
    """
    First-pass OCR for handwritten homework images or PDFs.

    Returns raw Mathpix JSON so we can inspect what works best
    before building the review workflow.
    """
    if not app_id or not app_key:
        raise MathpixError("Missing Mathpix credentials.")

    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "src": _to_data_url(file_path),
        "math_inline_delimiters": ["$", "$"],
        "rm_spaces": True,
        "include_latex": True,
        "include_line_data": include_line_data,
        "formats": ["text", "data", "html"],
    }

    resp = requests.post(MATHPIX_TEXT_URL, headers=headers, json=payload, timeout=120)

    try:
        data = resp.json()
    except Exception as e:
        raise MathpixError(f"Mathpix returned non-JSON response: {e}") from e

    if resp.status_code != 200:
        err = data.get("error") or data.get("error_info") or data
        raise MathpixError(f"Mathpix OCR failed: {err}")

    if data.get("error"):
        raise MathpixError(f"Mathpix OCR error: {data['error']}")

    return data