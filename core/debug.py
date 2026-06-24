# core/debug.py
from __future__ import annotations

import json
import re
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEBUG_DIR, RUNS_ROOT, env_bool


def write_debug_log(prefix: str, exc: Exception) -> str:
    """Write traceback to a timestamped file, return the path string."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = DEBUG_DIR / f"{prefix}_{ts}.log"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    p.write_text(tb, encoding="utf-8")
    return str(p)


_TRACE_ENABLED = env_bool("MATHGRADE_DEBUG_TRACE", "1")
_TRACE_ROOT = Path(
    # Keep traces under RUNS_ROOT by default so they follow the existing runtime data root.
    # Override with MATHGRADE_DEBUG_TRACE_DIR if you want them elsewhere.
    __import__("os").getenv("MATHGRADE_DEBUG_TRACE_DIR") or (RUNS_ROOT / "debug_traces")
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_segment(value: Any, fallback: str = "unknown", limit: int = 80) -> str:
    s = str(value or "").strip()
    if not s:
        s = fallback
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or fallback)[:limit]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-safe values for debug logging."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    return str(value)


class DebugTrace:
    """
    Tiny per-run debug trace.

    It is intentionally filesystem-first and UI-agnostic:
      debug_traces/{run_id}/
        run_log.jsonl
        manifest.json
        artifacts/...

    All write methods are best-effort and must never break grading/OCR.
    """

    def __init__(self, *, kind: str, metadata: dict[str, Any] | None = None, enabled: bool = True):
        self.enabled = bool(enabled and _TRACE_ENABLED)
        self.kind = _safe_segment(kind, "run")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:8]
        exam = _safe_segment((metadata or {}).get("exam_id"), "no_exam", 40)
        self.run_id = f"{ts}_{self.kind}_{exam}_{short}"
        self.path = _TRACE_ROOT / self.run_id
        self.artifacts_dir = self.path / "artifacts"
        self.log_path = self.path / "run_log.jsonl"
        self._counter = 0

        if not self.enabled:
            return

        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": self.run_id,
                "kind": kind,
                "created_at": _now_iso(),
                "trace_dir": str(self.path),
                "metadata": _jsonable(metadata or {}),
            }
            (self.path / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.log("trace", "created", status="ok", **(metadata or {}))
        except Exception:
            self.enabled = False

    def _artifact_path(self, name: str) -> Path:
        name = str(name or "artifact.txt").replace("\\", "/").lstrip("/")
        safe_parts = [_safe_segment(part, "part", 120) for part in name.split("/") if part]
        if not safe_parts:
            safe_parts = ["artifact.txt"]
        p = self.artifacts_dir.joinpath(*safe_parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _next_name(self, name: str) -> str:
        self._counter += 1
        return f"{self._counter:02d}_{name}"

    def log(self, stage: str, event: str, status: str = "ok", **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            payload = {
                "ts": _now_iso(),
                "run_id": self.run_id,
                "kind": self.kind,
                "stage": stage,
                "event": event,
                "status": status,
                **_jsonable(fields),
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_text(self, name: str, text: str | bytes | None, *, stage: str = "artifact") -> str | None:
        if not self.enabled:
            return None
        try:
            p = self._artifact_path(self._next_name(name))
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            p.write_text(text or "", encoding="utf-8")
            self.log(stage, "artifact_saved", path=str(p), bytes=p.stat().st_size)
            return str(p)
        except Exception:
            return None

    def save_bytes(self, name: str, data: bytes | None, *, stage: str = "artifact") -> str | None:
        if not self.enabled:
            return None
        try:
            p = self._artifact_path(self._next_name(name))
            p.write_bytes(data or b"")
            self.log(stage, "artifact_saved", path=str(p), bytes=p.stat().st_size)
            return str(p)
        except Exception:
            return None

    def save_json(self, name: str, data: Any, *, stage: str = "artifact") -> str | None:
        if not self.enabled:
            return None
        try:
            p = self._artifact_path(self._next_name(name))
            p.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(stage, "artifact_saved", path=str(p), bytes=p.stat().st_size)
            return str(p)
        except Exception:
            return None

    def save_file(self, source: str | Path | None, name: str | None = None, *, stage: str = "artifact") -> str | None:
        if not self.enabled or not source:
            return None
        try:
            src = Path(source)
            if not src.exists() or not src.is_file():
                self.log(stage, "artifact_missing", status="warning", source=str(src))
                return None
            p = self._artifact_path(self._next_name(name or src.name))
            shutil.copy2(src, p)
            self.log(stage, "artifact_saved", path=str(p), source=str(src), bytes=p.stat().st_size)
            return str(p)
        except Exception:
            return None

    def error(self, stage: str, exc: Exception, **fields: Any) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log(stage, "error", status="error", error=str(exc), **fields)
        self.save_text(f"{stage}_error_traceback.txt", tb, stage=stage)


class NoopTrace(DebugTrace):
    def __init__(self) -> None:
        self.enabled = False
        self.kind = "noop"
        self.run_id = ""
        self.path = Path("")
        self.artifacts_dir = Path("")
        self.log_path = Path("")
        self._counter = 0


def create_debug_trace(kind: str, **metadata: Any) -> DebugTrace:
    """Create a debug trace. Returns a disabled trace if creation fails."""
    try:
        return DebugTrace(kind=kind, metadata=metadata, enabled=True)
    except Exception:
        return NoopTrace()


def validate_bundle_snapshot(bundle: Any, *, bundle_kind: str = "bundle") -> dict[str, Any]:
    """
    Lightweight schema-agnostic validation for the files we debug most often.

    This does not replace Pydantic validation. It is a human-readable smoke test
    that helps identify the first stage where OCR/AI output became suspicious.
    """
    data = _jsonable(bundle)
    if not isinstance(data, dict):
        return {
            "bundle_kind": bundle_kind,
            "status": "error",
            "warnings": ["Bundle is not a JSON object."],
        }

    warnings: list[str] = []
    questions = data.get("questions") or []
    if not isinstance(questions, list) or not questions:
        warnings.append("No questions found.")
        questions = []

    seen: set[tuple[str, str]] = set()
    duplicate_keys: list[str] = []
    empty_question_text: list[str] = []
    empty_solution: list[str] = []
    review_items: list[str] = []
    part_count = 0

    expects_question_text = bundle_kind in {"questions_only", "questions_answers", "full_solution", "reference"}
    expects_solution = bundle_kind in {"answers_only", "questions_answers", "full_solution", "reference"}

    for q in questions:
        if not isinstance(q, dict):
            warnings.append("Question entry is not an object.")
            continue
        qid = str(q.get("question_id") or "?")
        parts = q.get("parts") or []
        if not isinstance(parts, list) or not parts:
            warnings.append(f"Question {qid} has no parts.")
            continue
        for p in parts:
            if not isinstance(p, dict):
                warnings.append(f"Question {qid} has a non-object part.")
                continue
            part_count += 1
            part = str(p.get("part") or p.get("part_key") or "").strip()
            key = (qid, part)
            key_label = f"Q{qid}{part}"
            if key in seen:
                duplicate_keys.append(key_label)
            seen.add(key)

            if expects_question_text and not str(p.get("question_text") or "").strip():
                empty_question_text.append(key_label)

            if expects_solution:
                has_solution = bool(str(p.get("official_solution") or "").strip() or str(p.get("expected_answer") or "").strip())
                if not has_solution:
                    empty_solution.append(key_label)

            if str(p.get("review_status") or "ok") != "ok" or p.get("warnings"):
                review_items.append(key_label)

    if duplicate_keys:
        warnings.append(f"Duplicate question/part keys: {', '.join(duplicate_keys[:20])}")
    if empty_question_text:
        warnings.append(f"Missing question_text: {', '.join(empty_question_text[:20])}")
    if empty_solution:
        warnings.append(f"Missing official_solution/expected_answer: {', '.join(empty_solution[:20])}")
    if review_items:
        warnings.append(f"Parts needing review or carrying warnings: {', '.join(review_items[:20])}")

    top_warnings = data.get("warnings") or []
    if top_warnings:
        warnings.append(f"Bundle-level warnings: {len(top_warnings)}")

    corrections = data.get("structure_corrections") or []
    if corrections:
        warnings.append(f"Structure corrections: {len(corrections)}")

    return {
        "bundle_kind": bundle_kind,
        "status": "warning" if warnings else "ok",
        "question_count": len(questions),
        "part_count": part_count,
        "duplicate_key_count": len(duplicate_keys),
        "empty_question_text_count": len(empty_question_text),
        "empty_solution_count": len(empty_solution),
        "review_item_count": len(review_items),
        "bundle_warning_count": len(top_warnings),
        "structure_correction_count": len(corrections),
        "warnings": warnings,
    }
