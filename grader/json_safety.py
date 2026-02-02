# -*- coding: utf-8 -*-
# grader/json_safety.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

# JSON escapes allowed inside strings
VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_first_json_value(text: str) -> Optional[str]:
    """
    Extract the first complete JSON value (object/array) from a larger string.
    This is brace/quote aware and ignores braces inside strings.
    Returns None if not found.
    """
    if not text:
        return None

    # find the first { or [
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch == "{":
            start, opener = i, "{"
            break
        if ch == "[":
            start, opener = i, "["
            break
    if start is None:
        return None

    closer = "}" if opener == "{" else "]"
    stack = [closer]

    in_str = False
    esc = False

    for j in range(start + 1, len(text)):
        ch = text[j]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : j + 1]

    # Not closed -> return None (we'll attempt close_truncated_json later)
    return None


def repair_invalid_unicode_escapes(s: str) -> str:
    r"""
    Replace occurrences of backslash + 'u' that are NOT followed by 4 hex digits
    with double-backslash + 'u' so it's treated as a literal sequence in JSON.
    Example: "\u12" -> "\\u12"
    """
    out = []
    i = 0
    n = len(s)
    hexd = "0123456789abcdefABCDEF"

    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] == "u":
            if i + 6 <= n and all(c in hexd for c in s[i + 2 : i + 6]):
                out.append("\\u" + s[i + 2 : i + 6])
                i += 6
            else:
                out.append("\\\\u")
                i += 2
        else:
            out.append(s[i])
            i += 1

    return "".join(out)


def repair_invalid_backslash_escapes_inside_strings(s: str) -> str:
    r"""
    Escape invalid backslash sequences inside JSON strings.
    Example: "\(" becomes "\\(" so JSON parsing succeeds.
    """
    out = []
    in_str = False
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Toggle string state when encountering an unescaped quote
        if ch == '"':
            bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                in_str = not in_str
            out.append(ch)
            i += 1
            continue

        if in_str and ch == "\\":
            # If backslash at end, escape it
            if i + 1 >= n:
                out.append("\\\\")
                i += 1
                continue

            nxt = s[i + 1]
            # Valid JSON escape -> keep
            if nxt in VALID_JSON_ESCAPES:
                out.append("\\")
                out.append(nxt)
                i += 2
            else:
                # Invalid escape -> double the backslash, keep next char as-is
                out.append("\\\\")
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def repair_unescaped_newlines_in_strings(s: str) -> str:
    r"""
    JSON does not allow literal newlines inside a string.
    If the model returns something like:
      "feedback": "line1
                  line2"
    this repairs it into:
      "feedback": "line1\nline2"

    Note: This is a best-effort scanner.
    """
    out = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                # drop or convert to \n; easiest is drop
                continue
            out.append(ch)
            continue

        # not in string
        if ch == '"':
            in_str = True
            out.append(ch)
        else:
            out.append(ch)

    return "".join(out)


def remove_trailing_commas(s: str) -> str:
    # {"a":1,} -> {"a":1} and [1,2,] -> [1,2]
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def close_truncated_json(s: str) -> str:
    """
    Best-effort closure for truncated JSON:
    - If inside a string at EOF, append closing quote
    - Close any remaining { [ with matching } ]
    - Remove trailing commas before final closings
    """
    if not s:
        return s

    stack: list[str] = []
    in_str = False
    esc = False
    out = []

    for ch in s:
        out.append(ch)

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack and stack[-1] == ch:
                stack.pop()

    if in_str:
        out.append('"')

    candidate = "".join(out)
    candidate = remove_trailing_commas(candidate)

    if stack:
        candidate += "".join(reversed(stack))

    candidate = remove_trailing_commas(candidate)
    return candidate


def safe_json_load(raw: str, debug_path: Path | None = None) -> dict[str, Any]:
    r"""
    Robust JSON loader with multi-step repairs for common LLM failures:
    - invalid \u escapes
    - invalid backslash escapes inside strings
    - literal newlines inside strings
    - trailing commas
    - truncated JSON (missing closing quotes/brackets/braces)
    - extra pre/post text: extract_first_json_value
    """
    if debug_path is not None:
        debug_path.write_text(raw, encoding="utf-8", errors="replace")

    last_err: Exception | None = None
    candidate = (raw or "").strip()

    for attempt in range(8):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            return {"_json_root": obj}
        except json.JSONDecodeError as e:
            last_err = e

        if attempt == 0:
            candidate = repair_invalid_unicode_escapes(candidate)
        elif attempt == 1:
            candidate = repair_invalid_backslash_escapes_inside_strings(candidate)
        elif attempt == 2:
            candidate = repair_unescaped_newlines_in_strings(candidate)
        elif attempt == 3:
            extracted = extract_first_json_value(candidate)
            if extracted:
                candidate = extracted
        elif attempt == 4:
            candidate = remove_trailing_commas(candidate)
        elif attempt == 5:
            candidate = close_truncated_json(candidate)
        elif attempt == 6:
            candidate = candidate.strip()

    assert last_err is not None
    raise last_err
