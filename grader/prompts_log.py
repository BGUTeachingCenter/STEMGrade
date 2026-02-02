# grader/prompts_log.py
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def write_prompts_txt(
    out_path: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    input_tex_path: Path,
    latex_text: str,
    schema: dict[str, Any],
) -> None:
    """
    Writes a reproducible log of the run (model + prompts + schema hashes).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_hash = sha256_text(user_prompt)
    tex_hash = sha256_text(latex_text)
    schema_hash = sha256_text(json.dumps(schema, sort_keys=True, ensure_ascii=False))

    max_user_chars = 4000
    user_excerpt = user_prompt[:max_user_chars]
    truncated = "" if len(user_prompt) <= max_user_chars else (
        f"\n\n[TRUNCATED: showing first {max_user_chars} chars]\n"
    )

    text = (
        "=== GRADER RUN (Ollama) ===\n"
        f"timestamp: {now}\n"
        f"model: {model}\n"
        f"input_tex: {input_tex_path.name}\n"
        f"input_tex_bytes: {input_tex_path.stat().st_size}\n"
        f"input_tex_sha256: {tex_hash}\n"
        f"user_prompt_sha256: {user_hash}\n"
        f"schema_sha256: {schema_hash}\n"
        "\n"
        "=== SYSTEM PROMPT ===\n"
        f"{system_prompt}\n"
        "\n"
        "=== USER PROMPT (excerpt) ===\n"
        f"{user_excerpt}"
        f"{truncated}\n"
        "=== JSON SCHEMA ===\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    )

    out_path.write_text(text, encoding="utf-8")
