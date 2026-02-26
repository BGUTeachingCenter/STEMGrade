# grader/ai_grading/gpt_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe_json_loads(s: str) -> dict:
    """
    Similar to your Ollama helper:
    - strips control chars
    - extracts the first {...} block if extra text exists
    """
    if not s:
        raise ValueError("Empty model response (expected JSON).")
    s2 = _CONTROL_CHARS_RE.sub("", s)
    start = s2.find("{")
    end = s2.rfind("}")
    candidate = s2[start : end + 1] if (start != -1 and end != -1 and end > start) else s2
    return json.loads(candidate)


def _extract_text_from_responses_api(resp_json: dict) -> str:
    """
    Responses API returns an 'output' array.
    We look for the first content item of type 'output_text' and return its 'text'.
    """
    out = resp_json.get("output")
    if isinstance(out, list):
        for item in out:
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        return c["text"]

    # Some SDK-like payloads also include output_text directly
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"]

    return ""


@dataclass
class GptClient:
    api_key: str
    model: str
    base_url: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # ✅ env reading ONLY here (mirrors your OllamaClient)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment (or pass api_key=...).")

        # Pick a sane default. Override via OPENAI_MODEL if you want.
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.15,
        timeout_s: int = 120,
        schema_name: str = "result",
        strict: bool = True,
    ) -> dict:
        """
        Returns a Python dict parsed from the model's structured JSON output.

        Uses Responses API + Structured Outputs (json_schema).
        """
        url = f"{self.base_url}/v1/responses"

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "response_format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": bool(strict),
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        # If OpenAI returns an error JSON, surface it clearly
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                raise RuntimeError(f"OpenAI request failed ({r.status_code}). Body:\n{r.text[:2000]}")
            raise RuntimeError(f"OpenAI request failed ({r.status_code}). Error:\n{json.dumps(err, indent=2)}")

        data = r.json()

        text = _extract_text_from_responses_api(data)
        if not text:
            # last-resort debug dump (trimmed)
            raise RuntimeError(
                "OpenAI returned no output_text. Response (trimmed):\n"
                + json.dumps(data, ensure_ascii=False, indent=2)[:2000]
            )

        # With json_schema, the model should output valid JSON matching schema,
        # but we keep your robust parser anyway.
        try:
            return _safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(
                f"OpenAI returned non-JSON content. First 500 chars:\n{text[:500]}"
            ) from e