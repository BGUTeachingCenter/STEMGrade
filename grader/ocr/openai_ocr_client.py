from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from grader.ai_clients.gpt_client import _make_requests_session


OPENAI_RESPONSES_URL_PATH = "/v1/responses"


class OpenAIOcrError(RuntimeError):
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
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_guess_mime(path)};base64,{b64}"


def _extract_output_text(resp_json: dict[str, Any]) -> str:
    """
    Extract text from the OpenAI Responses API response.
    Mirrors the logic used by your existing GPT client, but kept local so this
    OCR client can work independently.
    """
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"]

    output = resp_json.get("output")
    if isinstance(output, list):
        chunks: list[str] = []

        for item in output:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if not isinstance(content, list):
                continue

            for c in content:
                if not isinstance(c, dict):
                    continue

                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])

                # Some API variants return text in a plain "text" field.
                elif isinstance(c.get("text"), str):
                    chunks.append(c["text"])

        return "\n".join(x for x in chunks if x.strip()).strip()

    return ""


def process_image_or_pdf_with_openai(
    *,
    file_path: Path,
    api_key: str,
    model: str,
    base_url: str = "https://api.openai.com",
    timeout_s: int = 300,
) -> dict[str, Any]:
    """
    OCR-like extraction using OpenAI vision/file input.

    Returns a Mathpix-like dict shape:
      {
        "text": "...",
        "html": "",
        "latex_styled": "",
        "line_data": [],
        "_openai_mode": "image" | "pdf",
        ...
      }

    This keeps the downstream solution-bank flow almost unchanged.
    """
    api_key = (api_key or "").strip()
    model = (model or "").strip()
    base_url = (base_url or "https://api.openai.com").rstrip("/")

    if not api_key:
        raise OpenAIOcrError("Missing OPENAI_API_KEY in environment.")

    if not model:
        raise OpenAIOcrError("Missing OPENAI_OCR_MODEL / OPENAI_MODEL.")

    if not file_path.exists():
        raise OpenAIOcrError(f"OCR input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        raise OpenAIOcrError(f"Unsupported OpenAI OCR file type: {suffix}")

    data_url = _to_data_url(file_path)

    if suffix == ".pdf":
        file_item: dict[str, Any] = {
            "type": "input_file",
            "filename": file_path.name,
            "file_data": data_url,
        }
        mode = "pdf"
    else:
        file_item = {
            "type": "input_image",
            "image_url": data_url,
        }
        mode = "image"

    prompt = """
You are an OCR engine for Hebrew mathematics exams.

Extract the visible text from the uploaded file.

Requirements:
- Preserve Hebrew text.
- Preserve question numbers and part labels such as א, ב, ג.
- Preserve mathematical notation using LaTeX where useful.
- Preserve line breaks when they help structure.
- Do not summarize.
- Do not solve the exam.
- Do not add explanations.
- Return only the extracted OCR text.
""".strip()

    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    file_item,
                ],
            }
        ],
        "max_output_tokens": 12000,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    session = _make_requests_session()

    try:
        resp = session.post(
            f"{base_url}{OPENAI_RESPONSES_URL_PATH}",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
    except requests.RequestException as e:
        raise OpenAIOcrError(f"Could not reach OpenAI OCR API: {e}") from e

    if not resp.ok:
        try:
            err = resp.json()
            err_text = json.dumps(err, ensure_ascii=False, indent=2)
        except Exception:
            err_text = resp.text[:2000]
        raise OpenAIOcrError(f"OpenAI OCR failed ({resp.status_code}):\n{err_text}")

    data = resp.json()
    text = _extract_output_text(data)

    if not text.strip():
        raise OpenAIOcrError(
            "OpenAI OCR returned no text. Response preview:\n"
            + json.dumps(data, ensure_ascii=False, indent=2)[:2000]
        )

    return {
        "_openai_mode": mode,
        "model": model,
        "response_id": data.get("id"),
        "text": text,
        "html": "",
        "latex_styled": "",
        "line_data": [],
        "confidence": None,
        "is_handwritten": None,
    }