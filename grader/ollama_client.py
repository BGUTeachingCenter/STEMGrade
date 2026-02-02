# -*- coding: utf-8 -*-
# grader/ollama_client.py
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from grader.json_safety import extract_first_json_value, safe_json_load


def _post_ollama_generate(
    ollama_url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ollama_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def call_ollama_grade(
    *,
    ollama_url: str,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    timeout_seconds: int,
    raw_debug_path: Path | None = None,
    extracted_debug_path: Path | None = None,
) -> dict[str, Any]:
    # -------- First pass (streaming) --------
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "format": schema,
        "stream": True,
        "options": {"temperature": 0},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ollama_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        for raw_line in resp:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("response"):
                chunks.append(msg["response"])
            if msg.get("done") is True:
                break

    full = "".join(chunks).strip()
    if raw_debug_path is not None:
        raw_debug_path.write_text(full, encoding="utf-8", errors="replace")

    extracted = extract_first_json_value(full) or full
    if extracted_debug_path is not None:
        extracted_debug_path.write_text(extracted, encoding="utf-8", errors="replace")

    # Try to parse with repairs
    try:
        return safe_json_load(extracted, debug_path=None)
    except Exception:
        pass

    # -------- Second pass (repair via Ollama) --------
    repair_system = (
        "You are a strict JSON repair tool.\n"
        "Return ONLY valid JSON that conforms exactly to the provided schema.\n"
        "Do not add any commentary.\n"
    )

    repair_prompt = (
        "Fix the following into valid JSON that matches the schema.\n"
        "Return JSON only.\n\n"
        f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"BROKEN_OUTPUT:\n{extracted}\n"
    )

    payload2 = {
        "model": model,
        "system": repair_system,
        "prompt": repair_prompt,
        "format": schema,
        "stream": False,
        "options": {"temperature": 0},
    }

    resp2 = _post_ollama_generate(ollama_url, payload2, timeout_seconds)
    text2 = (resp2.get("response") or "").strip()

    # Sometimes non-stream returns extra text; extract JSON again
    extracted2 = extract_first_json_value(text2) or text2
    if extracted_debug_path is not None:
        # overwrite with repaired attempt for inspection
        extracted_debug_path.write_text(extracted2, encoding="utf-8", errors="replace")

    return safe_json_load(extracted2, debug_path=None)
