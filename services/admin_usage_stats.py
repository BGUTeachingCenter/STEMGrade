from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.ai_clients.ai_usage_logger import usage_log_dir


_EMPTY_USAGE = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "total_tokens": 0,
    "ocr_tokens": 0,
    "ai_tokens": 0,
}


def _empty_usage() -> dict[str, int]:
    return dict(_EMPTY_USAGE)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _add_usage(target: dict[str, int], rec: dict[str, Any]) -> None:
    total = _safe_int(rec.get("total_tokens"))
    input_tokens = _safe_int(rec.get("input_tokens") or rec.get("prompt_tokens"))
    output_tokens = _safe_int(rec.get("output_tokens") or rec.get("candidate_tokens"))
    reasoning_tokens = _safe_int(rec.get("reasoning_tokens"))

    target["calls"] += 1
    target["input_tokens"] += input_tokens
    target["output_tokens"] += output_tokens
    target["reasoning_tokens"] += reasoning_tokens
    target["total_tokens"] += total

    task = str(rec.get("task") or "").lower()
    if task == "ocr":
        target["ocr_tokens"] += total
    else:
        target["ai_tokens"] += total


def _iter_usage_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = usage_log_dir()

    for path in sorted(root.glob("ai_usage_*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                records.append(rec)

    return records


def usage_by_teacher_id() -> dict[str, dict[str, int]]:
    """
    Aggregate AI/OCR token usage by teacher_id.

    This relies on routes adding teacher_id into bind_usage_context.
    Logs created before that patch will not be attributed to a voucher.
    """
    totals: dict[str, dict[str, int]] = {}

    for rec in _iter_usage_records():
        teacher_id = str(rec.get("teacher_id") or "").strip()
        if not teacher_id:
            continue

        entry = totals.setdefault(teacher_id, _empty_usage())
        _add_usage(entry, rec)

    return totals


def attach_usage_to_teachers(teachers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = usage_by_teacher_id()
    out: list[dict[str, Any]] = []

    for teacher in teachers:
        item = dict(teacher)
        teacher_id = str(item.get("teacher_id") or "").strip()
        item["token_usage"] = totals.get(teacher_id, _empty_usage())
        out.append(item)

    return out


def attach_usage_to_vouchers(vouchers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = usage_by_teacher_id()
    out: list[dict[str, Any]] = []

    for voucher in vouchers:
        item = dict(voucher)
        teacher_id = str(
            item.get("used_by_teacher_id")
            or item.get("teacher_id")
            or ""
        ).strip()
        item["token_usage"] = totals.get(teacher_id, _empty_usage()) if teacher_id else _empty_usage()
        out.append(item)

    return out