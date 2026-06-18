# grader/ai_grading/google_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        usage = data.get("usageMetadata") or {}
        pt = int(usage.get("promptTokenCount") or 0)
        ct = int(usage.get("candidatesTokenCount") or 0)
        tt = int(usage.get("totalTokenCount") or (pt + ct) or 0)
        self.total_prompt_tokens += pt
        self.total_candidate_tokens += ct
        self.total_tokens += tt

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
