from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def usage_log_dir() -> Path:
    """
    Shared directory for all AI/OCR usage logs.

    Override with:
      AI_USAGE_LOG_DIR=/path/to/logs

    Default:
      data/ai_usage_logs
    """
    root = Path(os.getenv("AI_USAGE_LOG_DIR") or "data/ai_usage_logs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_ai_usage(record: dict[str, Any]) -> None:
    """
    Append one JSONL record per AI/OCR call.

    Logging must never break OCR/grading.
    """
    try:
        day = datetime.now().strftime("%Y-%m-%d")
        path = usage_log_dir() / f"ai_usage_{day}.jsonl"

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **record,
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass