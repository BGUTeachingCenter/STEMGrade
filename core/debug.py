# core/debug.py
from __future__ import annotations

import traceback
from datetime import datetime
from .config import DEBUG_DIR

def write_debug_log(prefix: str, exc: Exception) -> str:
    """Write traceback to a timestamped file, return the path string."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = DEBUG_DIR / f"{prefix}_{ts}.log"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    p.write_text(tb, encoding="utf-8")
    return str(p)