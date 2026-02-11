# json_sanitizer.py
from __future__ import annotations
import re
from typing import Any, Dict, Tuple, List

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def _clean_string(s: str) -> tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    if not s:
        return s, stats

    s2, n = _CONTROL.subn("", s)
    if n: stats["control_chars_removed"] = n

    # Avoid $$ completely (your rule A)
    s3, n = re.subn(r"\$\$(.*?)\$\$", r"\\[\1\\]", s2, flags=re.DOTALL)
    if n: stats["double_dollars_to_brackets"] = n
    s2 = s3

    # If the model ever uses raw $...$, normalize it too
    s3, n = re.subn(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", r"\\(\1\\)", s2, flags=re.DOTALL)
    if n: stats["single_dollars_to_parens"] = n
    s2 = s3

    # Neutralize bare '&' (text or math). In JSON stage we do the *safe* thing:
    # turn '&' into '\&' so it won't become an alignment tab in any context.
    s3, n = re.subn(r"(?<!\\)&", r"\\&", s2)
    if n: stats["ampersands_escaped"] = n
    s2 = s3

    return s2, stats


def sanitize_grades_json(obj: Any) -> Tuple[Any, Dict[str, int]]:
    """
    Walk the grades dict and sanitize strings so that TeX generation is less likely to break.
    Returns (sanitized_obj, stats).
    """
    stats_total: Dict[str, int] = {}

    def bump(stats: Dict[str, int]):
        for k, v in stats.items():
            stats_total[k] = stats_total.get(k, 0) + v

    def walk(x: Any) -> Any:
        if isinstance(x, str):
            s2, st = _clean_string(x)
            bump(st)
            return s2
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(obj), stats_total
