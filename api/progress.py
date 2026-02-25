# api/progress.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["progress"])

_lock = Lock()
_store: dict[str, dict[str, Any]] = {}

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def init_job(job_id: str) -> None:
    with _lock:
        _store[job_id] = {"steps": [], "seq": 0, "done": False, "error": None}

def push(job_id: str, msg: str) -> None:
    with _lock:
        job = _store.setdefault(job_id, {"steps": [], "seq": 0, "done": False, "error": None})
        job["seq"] += 1
        job["steps"].append({"seq": job["seq"], "ts": _ts(), "msg": msg})

def done(job_id: str) -> None:
    with _lock:
        job = _store.get(job_id)
        if job:
            job["done"] = True

def fail(job_id: str, error: str) -> None:
    with _lock:
        job = _store.setdefault(job_id, {"steps": [], "seq": 0, "done": False, "error": None})
        job["error"] = error
        job["done"] = True

@router.get("/progress/{job_id}")
def get_progress(job_id: str):
    with _lock:
        return _store.get(job_id, {"steps": [], "done": False, "error": None})