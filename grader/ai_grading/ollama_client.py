# grader/ai_grading/ollama_client.py
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


@dataclass
class OllamaClient:
    base_url: str
    model: str

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        # ✅ env reading ONLY here
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "gemma3:4b"

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.15,
        timeout_s: int = 120,
    ) -> dict:
        # NOTE: you already have this implemented — keep your existing request code.
        # The key point is: it uses self.base_url and self.model.
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,  # or however you currently pass schema to Ollama
            "options": {"temperature": float(temperature)},
            "stream": False,
        }

        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()

        content = (data.get("message") or {}).get("content") or ""
        try:
            return _safe_json_loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Ollama returned non-JSON content. First 500 chars:\n{content[:500]}"
            ) from e
