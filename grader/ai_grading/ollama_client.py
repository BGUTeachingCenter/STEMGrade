from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaClient:
    """
    Minimal client for Ollama's HTTP API.

    Uses /api/chat with optional structured output (JSON schema).
    Docs: https://docs.ollama.com/api/chat  (base URL is typically http://localhost:11434/api)
    """
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    timeout_s: int = 180

    def chat_json(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Send a chat request asking for STRICT JSON output matching `schema`.

        Returns the parsed JSON dict.
        Raises RuntimeError if output isn't valid JSON.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,  # Ollama supports JSON schema for structured output
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
            },
        }

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        # Ollama returns assistant message in data["message"]["content"]
        content = (data.get("message") or {}).get("content", "").strip()

        try:
            return json.loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Ollama returned non-JSON content. First 500 chars:\n{content[:500]}"
            ) from e
