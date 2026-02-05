# grader/latex_cleanup.py
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


# A small, conservative replacement map for characters that frequently cause
# XeLaTeX/fontspec to fail on some Windows setups (especially under RTL/Hebrew).
# Keep this list short and predictable; it is applied to the FULL .tex source.
_UNICODE_REPLACEMENTS: Dict[str, str] = {
    "⚠": "(!)",   # warning sign
    "✅": "[OK]",
    "❌": "[X]",
    "✔": "[OK]",
    "✗": "[X]",
    "•": "-",     # bullet
    "‣": "-",
    "∙": "*",
    "·": "*",
    "…": "...",
    "–": "-",     # en dash
    "—": "-",     # em dash
    "−": "-",     # minus sign (keep math robust)
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": "'",
    "‘": "'",
    "‹": "<",
    "›": ">",
    "«": "<<",
    "»": ">>",
    "\u00a0": " ",  # NBSP (kept as escape here; handled by normalize)
}

# Emoji blocks / pictographs: replace with empty by default (they often have no glyph).
# Ranges are inclusive.
_EMOJI_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport and Map
    (0x1F700, 0x1F77F),  # Alchemical Symbols
    (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
    (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x2600, 0x26FF),    # Misc symbols
    (0x2700, 0x27BF),    # Dingbats
)

# TeX is sensitive to some control chars. Keep newline + tab; drop the rest.
_ALLOWED_CONTROL = {"\n", "\t"}


def _is_emoji(cp: int) -> bool:
    for lo, hi in _EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def sanitize_unicode_for_xelatex(s: str, *, replace_emoji_with: str = "") -> str:
    """Best-effort Unicode cleanup for XeLaTeX.

    - normalizes newlines to \n
    - replaces a small set of problematic symbols with ASCII fallbacks
    - strips control characters (except \n and \t)
    - strips emoji/pictographs that commonly lack glyphs in typical fonts
    """
    if s is None:
        return ""

    # Normalize line endings early
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    out_chars: list[str] = []
    for ch in s:
        # Control characters (including NUL) can break TeX; drop them.
        if ord(ch) < 32 and ch not in _ALLOWED_CONTROL:
            continue

        # Replace a small known set.
        if ch in _UNICODE_REPLACEMENTS:
            out_chars.append(_UNICODE_REPLACEMENTS[ch])
            continue

        cp = ord(ch)
        if _is_emoji(cp):
            out_chars.append(replace_emoji_with)
            continue

        out_chars.append(ch)

    return "".join(out_chars)


_SETMAINFONT_RE = re.compile(r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})")


def force_setmainfont(tex: str, font_name: str) -> str:
    """Force \setmainfont{<font_name>} while preserving optional arguments."""
    if not tex:
        return tex
    return _SETMAINFONT_RE.sub(rf"\1{font_name}\3", tex)


def clean_tex_source(tex: str, *, font_name: str = "Arial") -> str:
    """Clean a full LaTeX document source before XeLaTeX compilation."""
    tex = sanitize_unicode_for_xelatex(tex)

    # Tabs -> spaces (tabs inside TeX can lead to surprising spacing)
    tex = tex.replace("\t", "  ")

    # Ensure requested font is used when present in the source.
    tex = force_setmainfont(tex, font_name)

    # Strip trailing whitespace on each line; ensure file ends with newline.
    tex = "\n".join(line.rstrip() for line in tex.split("\n")) + "\n"
    return tex
