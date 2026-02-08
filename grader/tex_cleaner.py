from __future__ import annotations

from dataclasses import dataclass, asdict
import re

# -----------------------
# Tokenization patterns
# -----------------------

# $$...$$
_DISPLAY_DOLLAR_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

# $...$  (avoid $$...$$ by negative lookaround)
_INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)

# \[...\]
_BRACKET_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

# \( ... \)
_PAREN_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

# \begin{equation}...\end{equation} etc.
_ENV_MATH_RE = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|alignat\*?)\}"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}",
    re.DOTALL,
)

# Detect $$$, $$$$, etc. (unescaped)
_MULTI_DOLLAR_RE = re.compile(r"(?<!\\)\${3,}")

# Fix invalid \left{ and \right} delimiters - detect set notation patterns
_SET_NOTATION_RE = re.compile(
    r"\\left\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}?\s*\\right\s*\}?",
    re.IGNORECASE | re.DOTALL
)

# Simpler cases of just \left{ or \right} without proper pairing
_INVALID_LEFT_BRACE_RE = re.compile(r"\\left\s*\{", re.IGNORECASE)
_INVALID_RIGHT_BRACE_RE = re.compile(r"\\right\s*\}", re.IGNORECASE)


@dataclass(frozen=True)
class CleanReport:
    changed: bool = False
    triple_dollars_fixed: int = 0
    bracket_math_converted: int = 0
    paren_math_converted: int = 0
    env_math_sanitized: int = 0
    dollar_math_sanitized: int = 0
    invalid_delimiters_fixed: int = 0
    set_notation_fixed: int = 0


def _normalize_line_endings(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _fix_triple_dollars(s: str) -> str:
    # $$$ -> $$, $$$$ -> $$, etc.
    return _MULTI_DOLLAR_RE.sub("$$", s)


def _convert_bracket_math(s: str) -> str:
    # Convert \[...\] => $$...$$ and \(..\) => $..$
    s = _BRACKET_DISPLAY_RE.sub(lambda m: "$$" + m.group(1) + "$$", s)
    s = _PAREN_INLINE_RE.sub(lambda m: "$" + m.group(1) + "$", s)
    return s


def _fix_set_notation(s: str) -> tuple[str, int]:
    """Fix set notation with invalid \left{ and \right} delimiters."""
    count = 0

    # First, handle complete set notation patterns like \left{\,\frac{k}{2^n}\ :\ k=0,1,2,\dots,2^n\,\right}.
    def replace_set_notation(match):
        content = match.group(1)
        # Use proper set notation with \{ and \}
        return r"\left\{" + content + r"\right\}"

    s, n1 = _SET_NOTATION_RE.subn(replace_set_notation, s)
    count += n1

    # Then handle any remaining isolated \left{ or \right}
    s, n2 = _INVALID_LEFT_BRACE_RE.subn(r"\\{", s)  # Convert to simple brace
    count += n2

    s, n3 = _INVALID_RIGHT_BRACE_RE.subn(r"\\}", s)  # Convert to simple brace
    count += n3

    return s, count


def _sanitize_math_inner(inner: str) -> str:
    # Avoid TeX comments breaking compilation inside math
    inner = inner.replace("%", r"\%")
    # Undo common text-escape artifact
    inner = inner.replace(r"\textbackslash{}", "\\")
    # Undo brace escaping inside math (but be careful not to break set notation)

    # First fix set notation issues
    inner, _ = _fix_set_notation(inner)

    return inner


def _sanitize_math_blocks(s: str) -> tuple[str, int]:
    # Sanitize $$...$$
    def repl_dd(m: re.Match) -> str:
        return "$$" + _sanitize_math_inner(m.group(1)) + "$$"

    # Sanitize $...$
    def repl_id(m: re.Match) -> str:
        return "$" + _sanitize_math_inner(m.group(1)) + "$"

    s, n1 = _DISPLAY_DOLLAR_RE.subn(repl_dd, s)
    s, n2 = _INLINE_DOLLAR_RE.subn(repl_id, s)
    return s, n1 + n2


def _sanitize_env_math(s: str) -> str:
    # Sanitize \begin{align}...\end{align} etc.
    def repl(m: re.Match) -> str:
        env = m.group("env")
        body = _sanitize_math_inner(m.group("body"))
        return rf"\begin{{{env}}}{body}\end{{{env}}}"

    return _ENV_MATH_RE.sub(repl, s)


def clean_tex(tex: str) -> tuple[str, dict]:
    """
    Robust TeX cleaning focused on preventing XeLaTeX crashes from AI output.
    Returns (cleaned_tex, report_dict).

    Performs:
      - normalize newlines
      - collapse $$$ / $$$$ -> $$
      - convert \[...\], \(..\) to $$..$$ / $..$
      - fix invalid \left{ and \right} delimiters (convert to proper set notation)
      - sanitize math blocks and common math environments
      - trim trailing whitespace
    """
    if tex is None:
        return "", asdict(CleanReport(changed=False))

    original = tex
    report = CleanReport(changed=False)

    # Normalize line endings
    tex = _normalize_line_endings(tex)

    # Fix set notation and invalid delimiters BEFORE processing math blocks
    tex2, n_set = _fix_set_notation(tex)
    report = CleanReport(**{**asdict(report), "set_notation_fixed": n_set})
    tex = tex2

    # $$$ -> $$
    tex2, n1 = _MULTI_DOLLAR_RE.subn("$$", tex)
    report = CleanReport(**{**asdict(report), "triple_dollars_fixed": n1})
    tex = tex2

    # \[...\] -> $$...$$
    tex2, n2 = _BRACKET_DISPLAY_RE.subn(lambda m: "$$" + m.group(1) + "$$", tex)
    report = CleanReport(**{**asdict(report), "bracket_math_converted": n2})
    tex = tex2

    # \(..\) -> $..$
    tex2, n3 = _PAREN_INLINE_RE.subn(lambda m: "$" + m.group(1) + "$", tex)
    report = CleanReport(**{**asdict(report), "paren_math_converted": n3})
    tex = tex2

    # env math sanitize
    tex2, n4 = _ENV_MATH_RE.subn(
        lambda m: rf"\begin{{{m.group('env')}}}{_sanitize_math_inner(m.group('body'))}\end{{{m.group('env')}}}",
        tex
    )
    report = CleanReport(**{**asdict(report), "env_math_sanitized": n4})
    tex = tex2

    # $$...$$ and $...$ sanitize
    tex, n5 = _sanitize_math_blocks(tex)
    report = CleanReport(**{**asdict(report), "dollar_math_sanitized": n5})

    # Strip trailing whitespace on each line; ensure file ends with newline
    tex = "\n".join(line.rstrip() for line in tex.splitlines()) + "\n"

    changed = (tex != original)
    report = CleanReport(**{**asdict(report), "changed": changed})
    return tex, asdict(report)


# Convenience function for backward compatibility
def clean_tex_simple(tex: str) -> str:
    """Simple version that returns only the cleaned tex without report."""
    cleaned, _ = clean_tex(tex)
    return cleaned