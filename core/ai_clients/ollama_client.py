# services/ai_grading/ollama_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from core.ai_clients.ai_usage_logger import log_ai_usage


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
        timeout_s: int = 600,
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

        prompt_eval_count = int(data.get("prompt_eval_count") or 0)
        eval_count = int(data.get("eval_count") or 0)
        total_tokens = prompt_eval_count + eval_count

        self.total_prompt_tokens = int(getattr(self, "total_prompt_tokens", 0) or 0) + prompt_eval_count
        self.total_candidate_tokens = int(getattr(self, "total_candidate_tokens", 0) or 0) + eval_count
        self.total_tokens = int(getattr(self, "total_tokens", 0) or 0) + total_tokens
        self.last_usage = {
            "input_tokens": prompt_eval_count,
            "output_tokens": eval_count,
            "total_tokens": total_tokens,
            "raw_usage": {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_duration": data.get("prompt_eval_duration"),
                "eval_duration": data.get("eval_duration"),
            },
        }

        log_ai_usage(
            {
                "task": "chat_json",
                "provider": "ollama",
                "model": self.model,
                "input_tokens": prompt_eval_count,
                "output_tokens": eval_count,
                "reasoning_tokens": 0,
                "total_tokens": total_tokens,
                "raw_usage": self.last_usage["raw_usage"],
            }
        )

        content = (data.get("message") or {}).get("content") or ""
        try:
            return _safe_json_loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Ollama returned non-JSON content. First 500 chars:\n{content[:500]}"
            ) from e
