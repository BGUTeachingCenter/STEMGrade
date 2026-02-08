from __future__ import annotations
import re

_HEB_TO_LAT = {
    "א": "a",
    "ב": "b",
    "ג": "c",
    "ד": "d",
    "ה": "e",
    "ו": "f",
    "ז": "g",
    "ח": "h",
    "ט": "i",
    "י": "j",
}

_LAT_OK = set("abcdefghijklmnopqrstuvwxyz")

def normalize_part(raw: str) -> str:
    """
    Canonicalize part labels to latin a,b,c... (or "" for none).
    Accepts: "a", "(a)", "א", "(א)", "A", " ב ", etc.
    """
    if raw is None:
        return ""

    s = str(raw).strip()
    if not s:
        return ""

    # remove surrounding parentheses/brackets and whitespace
    s = re.sub(r"^[\(\[\{]\s*", "", s)
    s = re.sub(r"\s*[\)\]\}]$", "", s)
    s = s.strip()

    if not s:
        return ""

    # If it's like "part a" / "סעיף א" you can optionally extract a single letter:
    # keep only first character if the token is longer
    ch = s[0]

    # Hebrew → latin
    if ch in _HEB_TO_LAT:
        return _HEB_TO_LAT[ch]

    # Latin
    ch = ch.lower()
    if ch in _LAT_OK:
        return ch

    # fallback: keep first char lowercased (but better to map explicitly)
    return ch
