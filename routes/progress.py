# routes/progress.py
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import APIRouter

from core.config import RUNS_ROOT

router = APIRouter(prefix="/routes", tags=["progress"])

_lock = Lock()

_PROGRESS_DIR = Path(RUNS_ROOT) / "progress"
_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

_SAFE_JOB_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,120}$")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _safe_job_id(job_id: str) -> str:
    job_id = (job_id or "").strip()
    if not _SAFE_JOB_RE.match(job_id):
        job_id = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", job_id)[:120] or "job"
    return job_id


def _path(job_id: str) -> Path:
    return _PROGRESS_DIR / f"{_safe_job_id(job_id)}.json"


def _read(job_id: str) -> Dict[str, Any]:
    p = _path(job_id)
    if not p.exists():
        return {"steps": [], "seq": 0, "done": False, "error": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # If the reader caught a partial write (rare), return safe default
        return {"steps": [], "seq": 0, "done": False, "error": "progress file read error"}


def _write_atomic(job_id: str, data: Dict[str, Any]) -> None:
    """
    Windows-safe atomic write:
    - write to a uniquely named temp file (same directory)
    - os.replace() into place
    - retry on PermissionError/WinError 5 (file temporarily locked by AV/indexer/reader)
    """
    p = _path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Unique temp name reduces collisions + helps AV/scan timing
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{time.time_ns()}")

    payload = json.dumps(data, ensure_ascii=False)

    # Write temp
    tmp.write_text(payload, encoding="utf-8")

    # Replace with retries (Windows can momentarily deny)
    last_err: Exception | None = None
    for _ in range(20):  # ~2 seconds total with sleep below
        try:
            os.replace(str(tmp), str(p))
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.10)
        except OSError as e:
            # Some WinError variants surface as OSError
            last_err = e
            time.sleep(0.10)

    # If still failing, try a non-atomic fallback (better than crashing the whole request)
    try:
        p.write_text(payload, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return
    except Exception:
        # Cleanup temp best-effort then re-raise the original error
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise last_err if last_err else RuntimeError("Failed writing progress file")


def init_job(job_id: str) -> None:
    with _lock:
        _write_atomic(job_id, {"steps": [], "seq": 0, "done": False, "error": None})


def push(job_id: str, msg: str) -> None:
    with _lock:
        data = _read(job_id)
        data["seq"] = int(data.get("seq") or 0) + 1
        steps = data.get("steps")
        if not isinstance(steps, list):
            steps = []
            data["steps"] = steps
        steps.append({"seq": data["seq"], "ts": _ts(), "msg": str(msg)})
        _write_atomic(job_id, data)


def done(job_id: str) -> None:
    with _lock:
        data = _read(job_id)
        data["done"] = True
        _write_atomic(job_id, data)


def fail(job_id: str, error: str) -> None:
    with _lock:
        data = _read(job_id)
        data["error"] = str(error)
        data["done"] = True
        _write_atomic(job_id, data)


@router.get("/progress/{job_id}")
def get_progress(job_id: str):
    # Reads are cheap; still use lock to avoid reading mid-write in the same process
    with _lock:
        return _read(job_id)