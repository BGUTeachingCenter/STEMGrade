from __future__ import annotations

import re

# Matches $...$ or $$...$$ so we can avoid touching inside math
_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

# Very common unicode math symbols -> LaTeX-ish
_UNICODE_REPL = {
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "→": r"\to",
    "∞": r"\infty",
    "×": r"\cdot",
    "·": r"\cdot",
    "∈": r"\in",
    "∉": r"\notin",
    "∪": r"\cup",
    "∩": r"\cap",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∑": r"\sum",
    "∫": r"\int",
    "∂": r"\partial",
    "∆": r"\Delta",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "λ": r"\lambda",
    "π": r"\pi",
    "θ": r"\theta",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
}

# Heuristic: fragments likely to be math (contains these tokens/patterns)
# - $$...$$ : starts with $$ not preceded by backslash
# - $...$   : starts with single $ not preceded by backslash, and not $$.
_MATH_TOKEN_RE = re.compile(
    r"(?<!\\)\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$",
    re.DOTALL
)


# Split long text into smaller chunks for wrapping (avoid wrapping whole paragraphs)
# We wrap only inside these fragments found by _MATHY_TOKEN_RE.
def _wrap_math_fragments(text: str) -> str:
    def repl(match: re.Match) -> str:
        frag = match.group(0)
        # Already wrapped?
        if frag.startswith("$") and frag.endswith("$"):
            return frag
        return f"${frag}$"

    # Wrap mathy tokens that aren't already within $..$ (we call this only on non-math segments)
    return _MATH_TOKEN_RE.sub(repl, text)


def normalize_math_text(s: str) -> str:
    """
    Best-effort normalization:
    - Replace unicode math symbols with LaTeX-ish tokens
    - In non-math text regions, wrap mathy fragments with $...$
    - Leaves existing $...$ / $$...$$ regions untouched
    """
    if not s:
        return s

    # normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    parts = _MATH_BLOCK_RE.split(s)
    out = []

    for part in parts:
        if not part:
            continue

        # If it's an existing math block, only do unicode replacements inside (safe),
        # but DO NOT add extra $ wrappers.
        if (part.startswith("$") and part.endswith("$")):
            inner = part[2:-2] if part.startswith("$$") else part[1:-1]
            for k, v in _UNICODE_REPL.items():
                inner = inner.replace(k, v)
            if part.startswith("$$"):
                out.append("$$" + inner + "$$")
            else:
                out.append("$" + inner + "$")
            continue

        # Non-math region: replace unicode symbols and wrap math-like fragments
        t = part
        for k, v in _UNICODE_REPL.items():
            t = t.replace(k, v)

        # common plain-text "delta(Δ)" etc.
        t = re.sub(r"\bdelta\s*\(", r"\\delta(", t)
        t = re.sub(r"\bepsilon\b", r"\\epsilon", t)
        t = re.sub(r"\blambda\s*\(", r"\\lambda(", t)
        t = re.sub(r"\bDelta\b", r"\\Delta", t)

        t = _wrap_math_fragments(t)
        out.append(t)

    return "".join(out)
