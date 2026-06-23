from __future__ import annotations

import contextvars
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


# Ambient attribution context (e.g. exam_id, content_type) merged into every
# usage record logged while the context is active. The AI/OCR clients are
# generic and do not know which solution a call belongs to, so the caller
# (e.g. the solution-bank upload route) sets this context around its work.
#
# contextvars are copied into worker threads by anyio's run_in_threadpool and
# into child asyncio tasks, so OCR calls dispatched to a thread still inherit
# the attribution set on the request coroutine.
_usage_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "ai_usage_context", default={}
)


def bind_usage_context(**fields: Any) -> None:
    """
    Set attribution fields for the rest of the current task/request context.

    Unlike ``ai_usage_context`` there is no reset: this is meant for a
    request handler whose contextvars copy is discarded when the request
    ends, so the attribution never leaks to other requests.
    """
    base = dict(_usage_context.get() or {})
    base.update({k: v for k, v in fields.items() if v is not None})
    _usage_context.set(base)


@contextmanager
def ai_usage_context(**fields: Any) -> Iterator[None]:
    """
    Attach attribution fields (exam_id, content_type, ...) to every usage
    record logged inside this block. None values are ignored.
    """
    base = dict(_usage_context.get() or {})
    base.update({k: v for k, v in fields.items() if v is not None})
    token = _usage_context.set(base)
    try:
        yield
    finally:
        _usage_context.reset(token)


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

        ctx = _usage_context.get() or {}

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            # Ambient attribution (exam_id, content_type) comes first so an
            # explicit field in `record` always wins on conflict.
            **ctx,
            **record,
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass