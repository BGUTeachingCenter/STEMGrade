from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

from core.ai_clients.ai_usage_logger import log_ai_usage
from schemas.ocr_response import AiUsage, OcrOptions, OcrPage, OcrResponse, guess_input_kind


MATHPIX_TEXT_URL = "https://api.mathpix.com/v3/text"
MATHPIX_PDF_URL = "https://api.mathpix.com/v3/pdf"


class MathpixOcrClientError(RuntimeError):
    pass


def _headers(app_id: str, app_key: str) -> dict[str, str]:
    return {
        "app_id": app_id,
        "app_key": app_key,
    }


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
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_guess_mime(path)};base64,{b64}"


def _json_or_error(resp: requests.Response, context: str) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception as e:
        raise MathpixOcrClientError(f"{context}: Mathpix returned non-JSON response: {e}") from e

    if resp.status_code >= 400:
        err = data.get("error") or data.get("error_info") or data
        raise MathpixOcrClientError(f"{context}: Mathpix failed: {err}")

    if data.get("error"):
        raise MathpixOcrClientError(f"{context}: Mathpix error: {data['error']}")

    return data


def _process_image(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    options: OcrOptions,
) -> OcrResponse:
    headers = {
        **_headers(app_id, app_key),
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "src": _to_data_url(file_path),
        "math_inline_delimiters": ["$", "$"],
        "rm_spaces": True,
        "include_latex": True,
        "include_line_data": bool(options.include_line_data),
        "formats": ["text", "data", "html", "latex_styled"],
        "data_options": {"include_latex": True},
        "metadata": {
            "improve_mathpix": False,
            **(options.extra.get("metadata") or {}),
        },
    }

    # Escape hatch for experiments.
    for k, v in (options.extra.get("mathpix_payload") or {}).items():
        payload[k] = v

    try:
        resp = requests.post(
            MATHPIX_TEXT_URL,
            headers=headers,
            json=payload,
            timeout=options.timeout_s,
        )
    except requests.RequestException as e:
        raise MathpixOcrClientError(f"Could not reach Mathpix image OCR API: {e}") from e

    data = _json_or_error(resp, "Image OCR")
    text = data.get("text", "") or ""

    return OcrResponse(
        provider="mathpix",
        model="mathpix",
        input_kind=guess_input_kind(file_path),
        source_filename=file_path.name,
        source_path=str(file_path),
        text=text,
        pages=[
            OcrPage(
                page_index=0,
                page_number=1,
                text=text,
                html=data.get("html", "") or "",
                latex_styled=data.get("latex_styled", "") or "",
                confidence=data.get("confidence"),
                line_data=data.get("line_data", []) or [],
            )
        ],
        html=data.get("html", "") or "",
        latex_styled=data.get("latex_styled", "") or "",
        confidence=data.get("confidence"),
        is_handwritten=data.get("is_handwritten"),
        provider_mode="image",
        usage=AiUsage(),
        raw=data,
    )


def _submit_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    options: OcrOptions,
) -> str:
    mathpix_options: dict[str, Any] = {
        "rm_spaces": True,
        "rm_fonts": True,
        "conversion_formats": {
            "tex.zip": True,
        },
        "metadata": {
            "improve_mathpix": False,
            **(options.extra.get("metadata") or {}),
        },
    }

    for k, v in (options.extra.get("mathpix_pdf_options") or {}).items():
        mathpix_options[k] = v

    try:
        with file_path.open("rb") as f:
            resp = requests.post(
                MATHPIX_PDF_URL,
                headers=_headers(app_id, app_key),
                files={"file": (file_path.name, f, "application/pdf")},
                data={"options_json": json.dumps(mathpix_options)},
                timeout=options.timeout_s,
            )
    except requests.RequestException as e:
        raise MathpixOcrClientError(f"Could not submit PDF to Mathpix: {e}") from e

    data = _json_or_error(resp, "PDF submit")
    pdf_id = data.get("pdf_id")

    if not pdf_id:
        raise MathpixOcrClientError(f"PDF submit did not return pdf_id: {data}")

    return str(pdf_id)


def _poll_pdf_until_done(
    *,
    pdf_id: str,
    app_id: str,
    app_key: str,
    timeout_s: int,
    poll_interval_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{MATHPIX_PDF_URL}/{pdf_id}",
                headers=_headers(app_id, app_key),
                timeout=60,
            )
        except requests.RequestException as e:
            raise MathpixOcrClientError(f"Could not poll Mathpix PDF status: {e}") from e

        data = _json_or_error(resp, "PDF status")
        last_status = data

        status = str(data.get("status") or "").lower()

        if status == "completed":
            return data

        if status in {"error", "failed"}:
            raise MathpixOcrClientError(f"Mathpix PDF OCR failed: {data}")

        time.sleep(poll_interval_s)

    raise MathpixOcrClientError(f"Mathpix PDF OCR timed out. Last status: {last_status}")


def _download_pdf_text(
    *,
    pdf_id: str,
    app_id: str,
    app_key: str,
) -> tuple[str, str]:
    last_error = ""

    for ext in ("mmd", "md"):
        try:
            resp = requests.get(
                f"{MATHPIX_PDF_URL}/{pdf_id}.{ext}",
                headers=_headers(app_id, app_key),
                timeout=120,
            )
        except requests.RequestException as e:
            last_error = str(e)
            continue

        if resp.status_code == 200 and resp.content.strip():
            text = resp.content.decode("utf-8", errors="replace")
            return text, ext

        last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"

    raise MathpixOcrClientError(f"Could not download Mathpix PDF text output. Last error: {last_error}")


def _process_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    options: OcrOptions,
) -> OcrResponse:
    pdf_id = _submit_pdf(
        file_path=file_path,
        app_id=app_id,
        app_key=app_key,
        options=options,
    )

    status = _poll_pdf_until_done(
        pdf_id=pdf_id,
        app_id=app_id,
        app_key=app_key,
        timeout_s=options.timeout_s,
    )

    text, ext = _download_pdf_text(
        pdf_id=pdf_id,
        app_id=app_id,
        app_key=app_key,
    )

    raw = {
        "pdf_id": pdf_id,
        "status": status,
        "downloaded_ext": ext,
        "text": text,
    }

    return OcrResponse(
        provider="mathpix",
        model="mathpix",
        input_kind="pdf",
        source_filename=file_path.name,
        source_path=str(file_path),
        text=text,
        pages=[
            OcrPage(
                page_index=0,
                page_number=1,
                text=text,
            )
        ],
        provider_mode="pdf",
        provider_document_id=pdf_id,
        provider_status=status,
        is_handwritten=True,
        usage=AiUsage(),
        raw=raw,
    )


def run_mathpix_ocr(
    *,
    file_path: Path,
    model: Optional[str],
    options: OcrOptions,
) -> OcrResponse:
    app_id = (os.getenv("MATHPIX_APP_ID") or "").strip()
    app_key = (os.getenv("MATHPIX_APP_KEY") or "").strip()

    if not app_id or not app_key:
        raise MathpixOcrClientError("Missing MATHPIX_APP_ID / MATHPIX_APP_KEY.")

    if not file_path.exists():
        raise MathpixOcrClientError(f"OCR input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        result = _process_pdf(
            file_path=file_path,
            app_id=app_id,
            app_key=app_key,
            options=options,
        )

    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        result = _process_image(
            file_path=file_path,
            app_id=app_id,
            app_key=app_key,
            options=options,
        )

    else:
        raise MathpixOcrClientError(f"Unsupported Mathpix OCR file type: {suffix}")

    log_ai_usage(
        {
            "task": "ocr",
            "provider": result.provider,
            "model": result.model,
            "source_filename": result.source_filename,
            "input_kind": result.input_kind,
            "provider_mode": result.provider_mode,
            "provider_document_id": result.provider_document_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    )

    return result