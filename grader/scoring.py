# grader/scoring.py
from __future__ import annotations

from typing import Any


def compute_total(items: list[dict[str, Any]]) -> int:
    """Compute average score across items. Returns 0 if no items."""
    if not items:
        return 0

    scores: list[int] = []
    for it in items:
        try:
            scores.append(int(it.get("score", 0)))
        except (TypeError, ValueError):
            scores.append(0)

    return round(sum(scores) / len(scores))
