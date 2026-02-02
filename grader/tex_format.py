# grader/tex_format.py
from __future__ import annotations

import re

# --- TeX escaping (text-mode) ---
_TEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Identify math spans \( ... \) so we DO NOT wrap them with \LR{...}
_MATH_SPAN_RE = re.compile(r"(\\\(.+?\\\))", re.DOTALL)

# LTR tokens to wrap; keep it conservative
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/\-:+*=(){}\[\]]*[A-Za-z0-9]+)*")

# Replace a few unicode math glyphs that often become squares in PDFs
UNICODE_MATH_TO_LATEX = {
    "Σ": r"\sum",
    "∑": r"\sum",
    "∫": r"\int",
    "∞": r"\infty",
    "π": r"\pi",
    "Π": r"\Pi",
    "λ": r"\lambda",
    "μ": r"\mu",
    "θ": r"\theta",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "ε": r"\varepsilon",
    "δ": r"\delta",
    "Δ": r"\Delta",
    "∈": r"\in",
    "≤": r"\le",
    "≥": r"\ge",
    "→": r"\to",
    "×": r"\cdot",
    "·": r"\cdot",
    "−": "-",  # unicode minus
}


def normalize_unicode_math(s: str) -> str:
    """Replace common unicode math symbols with LaTeX commands."""
    if not s:
        return ""
    for k, v in UNICODE_MATH_TO_LATEX.items():
        s = s.replace(k, v)
    return s


def escape_tex_plain(s: str) -> str:
    """Escape text for LaTeX (NOT math)."""
    if s is None:
        return ""
    return "".join(_TEX_ESCAPE_MAP.get(ch, ch) for ch in s)


def rtl_lr_mix_to_tex(s: str) -> str:
    r"""
    Convert a mixed Hebrew/LTR string into LaTeX-safe output.

    Rules:
    - Escape normal text.
    - Wrap LTR tokens in \LR{...} to prevent RTL reordering.
    - DO NOT touch math spans already written as \( ... \).
    - DO NOT add \( \) wrappers (that was causing \(\( ... )\)).
    """
    if not s:
        return ""

    s = normalize_unicode_math(s)

    out_lines: list[str] = []
    for line in s.splitlines():
        # Split by existing math spans \( ... \)
        parts = _MATH_SPAN_RE.split(line)

        rendered_parts: list[str] = []
        for part in parts:
            if not part:
                continue

            # If this is a math span, keep it as-is (but do NOT escape it)
            if part.startswith(r"\(") and part.endswith(r"\)"):
                rendered_parts.append(part)
                continue

            # Otherwise: text segment. Escape + wrap LTR tokens with \LR{...}
            idx = 0
            text = part
            segs: list[str] = []
            for m in _LATIN_TOKEN_RE.finditer(text):
                a, b = m.start(), m.end()
                if a > idx:
                    segs.append(escape_tex_plain(text[idx:a]))
                token = escape_tex_plain(m.group(0))
                segs.append(r"\LR{" + token + "}")
                idx = b
            if idx < len(text):
                segs.append(escape_tex_plain(text[idx:]))

            rendered_parts.append("".join(segs))

        out_lines.append("".join(rendered_parts))

    return (r"\\ " + "\n").join(out_lines)
