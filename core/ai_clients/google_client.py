# services/ai_grading/google_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
import base64
from pathlib import Path

from core.ai_clients.ai_usage_logger import log_ai_usage
from schemas.ocr_response import AiUsage, OcrOptions, OcrPage, OcrResponse, guess_input_kind

import requests


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe_json_loads(s: str) -> dict:
    if not s:
        raise ValueError("Empty model response (expected JSON).")

    s2 = _CONTROL_CHARS_RE.sub("", s)

    start = s2.find("{")
    end = s2.rfind("}")
    candidate = s2[start : end + 1] if (start != -1 and end != -1 and end > start) else s2
    return json.loads(candidate)


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Gemini responseSchema is a restricted subset. Remove keys it rejects.
    DROP_KEYS = {
        "additionalProperties",
        "$schema", "$id", "definitions", "$defs",
        "patternProperties", "propertyNames",
        "dependencies", "dependentSchemas", "dependentRequired",
        "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
        "examples", "default", "format",
        "readOnly", "writeOnly", "nullable",
    }

    def rec(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items() if k not in DROP_KEYS}
        if isinstance(x, list):
            return [rec(i) for i in x]
        return x

    cleaned = rec(schema or {})
    if isinstance(cleaned, dict) and "type" not in cleaned:
        cleaned["type"] = "object"
    return cleaned


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


def _file_to_inline_data(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")

    return {
        "inline_data": {
            "mime_type": _guess_mime(path),
            "data": b64,
        }
    }


def _build_ocr_prompt(options: OcrOptions) -> str:
    return f"""
You are a neutral OCR engine for mathematics documents.

Extract the visible text from the uploaded file.

Requirements:
- Preserve the original language. Language hint: {options.language_hint}.
- Preserve Hebrew text when visible.
- Preserve question numbers and part labels such as א, ב, ג, a, b, c.
- Preserve mathematical notation using LaTeX where useful.
- Preserve line breaks when they help structure.
- Do not solve.
- Do not correct mathematical mistakes.
- Do not summarize.
- Do not add explanations.
- Return only the extracted OCR text.
""".strip()


def _extract_gemini_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []

    for cand in data.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)

    return "\n".join(chunks).strip()


def _extract_google_usage(data: dict[str, Any]) -> AiUsage:
    usage = data.get("usageMetadata") or {}

    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens) or 0)
    reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)

    return AiUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage,
    )

@dataclass
class GoogleClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model = (model or os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "").strip()
        self.api_base = (os.getenv("GOOGLE_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
        self.api_version = (os.getenv("GOOGLE_API_VERSION") or "v1beta").strip()

        # ✅ token counters (Gemini usageMetadata)
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_candidate_tokens = 0
        self.total_thoughts_tokens = 0
        self.last_usage: dict[str, Any] = {}


    def _record_usage(
        self,
        *,
        data: dict[str, Any],
        task: str,
        source_filename: str = "",
        input_kind: str = "",
        response_id: str = "",
    ) -> AiUsage:
        usage = _extract_google_usage(data)

        self.total_prompt_tokens += usage.input_tokens
        self.total_candidate_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_thoughts_tokens += usage.reasoning_tokens

        self.last_usage = usage.model_dump()

        log_ai_usage(
            {
                "task": task,
                "provider": "google",
                "model": self.model,
                "source_filename": source_filename,
                "input_kind": input_kind,
                "response_id": response_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "raw_usage": usage.raw_usage,
            }
        )

        return usage


    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.15,
        timeout_s: int = 120,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")
        if not self.model:
            raise RuntimeError("Missing GOOGLE_MODEL (or GEMINI_MODEL), or set client.model before calling.")

        url = f"{self.api_base}/{self.api_version}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        safe_schema = _sanitize_schema_for_gemini(schema)

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "responseMimeType": "application/json",
                "responseSchema": safe_schema,
            },
        }

        r = requests.post(url, params=params, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"Google Gemini API error {r.status_code}: {r.text[:1000]}")

        data = r.json()

        # ✅ accumulate usageMetadata if present
        self._record_usage(
            data=data,
            task="chat_json",
        )

        text = (
                   data.get("candidates", [{}])[0]
                   .get("content", {})
                   .get("parts", [{}])[0]
                   .get("text", "")
               ) or ""

        if not text:
            raise RuntimeError(
                "Gemini returned empty text. First 800 chars of response:\n"
                + json.dumps(data, ensure_ascii=False)[:800]
            )

        try:
            return _safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Gemini returned non-JSON content. First 500 chars:\n{text[:500]}"
            ) from e


    def ocr_document(
        self,
        *,
        file_path: Path,
        model: Optional[str] = None,
        options: Optional[OcrOptions] = None,
    ) -> OcrResponse:
        """
        OCR-like extraction using Gemini / Google AI Studio.

        This belongs here, not in ocr_google_client.py, because it uses the same
        Google API key, model config, endpoint config, and token logging.
        """
        options = options or OcrOptions()

        model_to_use = (
            model
            or os.getenv("GOOGLE_OCR_MODEL")
            or os.getenv("GEMINI_OCR_MODEL")
            or self.model
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("GEMINI_MODEL")
            or ""
        ).strip()

        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")

        if not model_to_use:
            raise RuntimeError("Missing GOOGLE_OCR_MODEL / GEMINI_OCR_MODEL / GOOGLE_MODEL / GEMINI_MODEL.")

        if not file_path.exists():
            raise RuntimeError(f"OCR input file does not exist: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
            raise RuntimeError(f"Unsupported Google OCR file type: {suffix}")

        url = f"{self.api_base}/{self.api_version}/models/{model_to_use}:generateContent"
        params = {"key": self.api_key}

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": _build_ocr_prompt(options)},
                        _file_to_inline_data(file_path),
                    ],
                }
            ],
            "generationConfig": {
                "temperature": float(options.temperature),
                "maxOutputTokens": int(options.max_output_tokens),
            },
        }

        # Escape hatch for OCR experiments.
        for k, v in (options.extra.get("google_payload") or {}).items():
            payload[k] = v

        r = requests.post(
            url,
            params=params,
            json=payload,
            timeout=options.timeout_s,
        )

        if r.status_code != 200:
            raise RuntimeError(f"Google Gemini OCR error {r.status_code}: {r.text[:2000]}")

        data = r.json()
        text = _extract_gemini_text(data)

        if not text.strip():
            raise RuntimeError(
                "Google Gemini OCR returned empty text. First 1000 chars of response:\n"
                + json.dumps(data, ensure_ascii=False)[:1000]
            )

        old_model = self.model
        self.model = model_to_use
        try:
            usage = self._record_usage(
                data=data,
                task="ocr",
                source_filename=file_path.name,
                input_kind=guess_input_kind(file_path),
            )
        finally:
            self.model = old_model

        mode = "pdf" if suffix == ".pdf" else "image"

        return OcrResponse(
            provider="google",
            model=model_to_use,
            input_kind=guess_input_kind(file_path),
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
            provider_mode=mode,
            provider_document_id=None,
            provider_status=None,
            response_id=None,
            usage=usage,
            raw=data,
        )