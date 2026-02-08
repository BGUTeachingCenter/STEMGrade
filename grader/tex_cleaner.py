"""
Extremely robust LaTeX cleaning for XeLaTeX compilation.
Handles all common issues with AI-generated LaTeX content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CleanupReport:
    changed: bool = False
    fixes_applied: Dict[str, int] = None

    def __post_init__(self):
        if self.fixes_applied is None:
            object.__setattr__(self, 'fixes_applied', {})


# =============================================
# REGEX PATTERNS
# =============================================

# Math delimiters
_MATH_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)
_BRACKET_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_PAREN_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

# Math environments
_MATH_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|alignat\*?)\}"
    r"(.*?)"
    r"\\end\{\1\}",
    re.DOTALL | re.IGNORECASE
)

# Problem patterns
_MULTI_DOLLAR_RE = re.compile(r"(?<!\\)\${3,}")
_TEXTBACKSLASH_RE = re.compile(r"\\textbackslash\{\}")
_TEXTBRACE_RE = re.compile(r"\\textbraceleft|\\textbraceright")

# Invalid LaTeX constructs
_INVALID_LEFT_BRACE_RE = re.compile(r"\\left\s*\{", re.IGNORECASE)
_INVALID_RIGHT_BRACE_RE = re.compile(r"\\right\s*\}", re.IGNORECASE)

# Escaped dollars in wrong contexts
_ESCAPED_DOLLAR_MATH_RE = re.compile(r"\\\$\\begin\{|\\\$\\end\{|\\\$\s*[a-zA-Z_\\]")

# Common undefined commands
_UNDEFINED_COMMANDS = {
    r'\\abs\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"|{m.group(1)}|",
    r'\\norm\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\|{m.group(1)}\\|",
    r'\\floor\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\lfloor {m.group(1)} \\rfloor",
    r'\\ceil\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\lceil {m.group(1)} \\rceil",
}

# Text escaping patterns
_TEXT_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Reverse map for unescaping in math
_MATH_UNESCAPE_MAP = {
    r"\textbackslash{}": "\\",
    r"\textasciicircum{}": "^",
    r"\textasciitilde{}": "~",
    r"\_": "_",
    r"\&": "&",
    r"\#": "#",
    r"\{": "{",
    r"\}": "}",
}


# =============================================
# CORE CLEANING FUNCTIONS
# =============================================

def _normalize_line_endings(text: str) -> str:
    """Normalize all line endings to Unix style."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _fix_textbackslash_sequences(text: str) -> Tuple[str, int]:
    """Fix problematic \\textbackslash{} and \\textbrace sequences."""
    count = 0

    # Fix \\textbackslash{} -> \\
    text, n1 = _TEXTBACKSLASH_RE.subn("\\\\", text)
    count += n1

    # Fix \\textbraceleft/\\textbraceright 
    text, n2 = _TEXTBRACE_RE.subn("", text)
    count += n2

    return text, count


def _fix_multi_dollars(text: str) -> Tuple[str, int]:
    """Convert $$$, $$$$, etc. to $$."""
    text, count = _MULTI_DOLLAR_RE.subn("$$", text)
    return text, count


def _fix_escaped_dollars_in_commands(text: str) -> Tuple[str, int]:
    """Fix \\$\\begin{} -> \\begin{} etc."""
    patterns = [
        (r"\\\$\\begin\{", r"\\begin{"),
        (r"\\\$\\end\{", r"\\end{"),
        (r"\\\$\s*(\\[a-zA-Z]+)", r"\1"),  # \\$ \\command -> \\command
    ]

    count = 0
    for pattern, replacement in patterns:
        text, n = re.subn(pattern, replacement, text)
        count += n

    return text, count


def _fix_invalid_delimiters(text: str) -> Tuple[str, int]:
    """Fix invalid \\left{ and \\right} delimiters."""
    count = 0

    # \\left{ -> \\{
    text, n1 = _INVALID_LEFT_BRACE_RE.subn(r"\\{", text)
    count += n1

    # \\right} -> \\}  
    text, n2 = _INVALID_RIGHT_BRACE_RE.subn(r"\\}", text)
    count += n2

    return text, count


def _fix_undefined_commands(text: str) -> Tuple[str, int]:
    """Replace undefined math commands with standard equivalents."""
    count = 0

    for pattern, replacement_func in _UNDEFINED_COMMANDS.items():
        text, n = re.subn(pattern, replacement_func, text)
        count += n

    return text, count


def _convert_bracket_math(text: str) -> Tuple[str, int]:
    """Convert \\[...\\] -> $$...$$ and \\(...\\) -> $...$."""
    count = 0

    # \\[...\\] -> $$...$$
    text, n1 = _BRACKET_DISPLAY_RE.subn(lambda m: "$$" + m.group(1) + "$$", text)
    count += n1

    # \\(...\\) -> $...$
    text, n2 = _PAREN_INLINE_RE.subn(lambda m: "$" + m.group(1) + "$", text)
    count += n2

    return text, count


def _clean_math_content(math_content: str) -> str:
    """Clean content that should be in math mode."""
    if not math_content:
        return math_content

    content = math_content

    # Fix undefined commands
    content, _ = _fix_undefined_commands(content)

    # Unescape text-mode artifacts
    for escaped, unescaped in _MATH_UNESCAPE_MAP.items():
        content = content.replace(escaped, unescaped)

    # Fix invalid delimiters
    content, _ = _fix_invalid_delimiters(content)

    # Escape % to prevent comments
    content = content.replace("%", r"\%")

    return content


def _sanitize_math_blocks(text: str) -> Tuple[str, int]:
    """Clean $$...$$ and $...$ blocks."""
    count = 0

    def clean_display_math(match):
        nonlocal count
        count += 1
        return "$$" + _clean_math_content(match.group(1)) + "$$"

    def clean_inline_math(match):
        nonlocal count
        count += 1
        return "$" + _clean_math_content(match.group(1)) + "$"

    # Clean display math
    text = _MATH_DISPLAY_RE.sub(clean_display_math, text)

    # Clean inline math  
    text = _MATH_INLINE_RE.sub(clean_inline_math, text)

    return text, count


def _sanitize_math_environments(text: str) -> Tuple[str, int]:
    """Clean math environments like \\begin{equation}...\\end{equation}."""

    def clean_env(match):
        env_name = match.group(1)
        content = _clean_math_content(match.group(2))
        return f"\\begin{{{env_name}}}{content}\\end{{{env_name}}}"

    text, count = _MATH_ENV_RE.subn(clean_env, text)
    return text, count


def _escape_text_safely(text: str) -> str:
    """Escape special characters in text mode, but preserve valid LaTeX."""
    if not text:
        return text

    # Don't escape text that's already in math or LaTeX environments
    if any(marker in text for marker in ["$$", "$", "\\begin{", "\\end{", "\\section", "\\item"]):
        return text

    # Only escape if it looks like plain text
    escaped = ""
    for char in text:
        escaped += _TEXT_ESCAPE_MAP.get(char, char)

    return escaped


def _fix_malformed_environments(text: str) -> Tuple[str, int]:
    """Fix malformed LaTeX environments and commands."""
    fixes = [
        # Fix \\begin\\{itemize\\} -> \\begin{itemize}
        (r"\\begin\\?\{([^}]+)\\?\}", r"\\begin{\1}"),
        (r"\\end\\?\{([^}]+)\\?\}", r"\\end{\1}"),

        # Fix double backslashes in commands
        (r"\\\\([a-zA-Z]+)", r"\\\1"),

        # Fix escaped braces in command names  
        (r"\\([a-zA-Z]+)\\?\{", r"\\\1{"),

        # Fix newlines in wrong places
        (r"\\newline\s*\\item", r"\\item"),
        (r"\\par\s*\\item", r"\\item"),
    ]

    count = 0
    for pattern, replacement in fixes:
        text, n = re.subn(pattern, replacement, text)
        count += n

    return text, count


def _final_cleanup(text: str) -> str:
    """Final cleanup pass."""
    # Strip trailing whitespace from each line
    lines = text.splitlines()
    lines = [line.rstrip() for line in lines]

    # Ensure document ends with newline
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"

    return result


def _preserve_unicode_text(text: str) -> str:
    """
    Ensure Unicode text (including Hebrew) is preserved during cleaning.
    Only modify LaTeX-specific issues, not content.
    """
    if not text:
        return text

    # Don't modify Hebrew characters or other Unicode content
    # Only fix LaTeX syntax issues

    # Hebrew characters range: \u0590-\u05FF
    # Arabic characters range: \u0600-\u06FF
    # We want to preserve these completely

    return text

# =============================================
# PUBLIC API  
# =============================================


def clean_tex_robust(tex: str, font_name: str = "Arial") -> Tuple[str, str]:
    """
    Extremely robust LaTeX cleaning for XeLaTeX compilation.

    Args:
        tex: Raw LaTeX content
        font_name: Font to use (for XeLaTeX)

    Returns:
        Tuple of (cleaned_tex, human_readable_report)
    """
    if not tex:
        return "", "No content to clean."

    original = tex  # FIX: Store original before processing

    # Ensure we preserve Unicode content
    tex = _preserve_unicode_text(tex)
    fixes = {}

    # Phase 1: Basic normalization
    tex = _normalize_line_endings(tex)

    # Phase 2: Fix dangerous sequences
    tex, n = _fix_textbackslash_sequences(tex)
    if n: fixes["textbackslash_fixed"] = n

    tex, n = _fix_escaped_dollars_in_commands(tex)
    if n: fixes["escaped_dollars_fixed"] = n

    tex, n = _fix_malformed_environments(tex)
    if n: fixes["malformed_environments_fixed"] = n

    # Phase 3: Math delimiter fixes
    tex, n = _fix_multi_dollars(tex)
    if n: fixes["multi_dollars_fixed"] = n

    tex, n = _convert_bracket_math(tex)
    if n: fixes["bracket_math_converted"] = n

    # Phase 4: Content cleaning
    tex, n = _fix_undefined_commands(tex)
    if n: fixes["undefined_commands_fixed"] = n

    tex, n = _fix_invalid_delimiters(tex)
    if n: fixes["invalid_delimiters_fixed"] = n

    # Phase 5: Math block sanitization
    tex, n = _sanitize_math_blocks(tex)
    if n: fixes["math_blocks_sanitized"] = n

    tex, n = _sanitize_math_environments(tex)
    if n: fixes["math_environments_sanitized"] = n

    # Phase 6: Font handling (XeLaTeX)
    tex = _handle_font_setup(tex, font_name)

    # Phase 7: Final cleanup
    tex = _final_cleanup(tex)

    # Generate report
    report_lines = ["LaTeX cleanup report:"]
    if not fixes:
        report_lines.append("  No issues found - document was already clean.")
    else:
        for fix_type, count in fixes.items():
            report_lines.append(f"  {fix_type}: {count}")

    changed = tex != original  # FIX: Now original is defined
    report_lines.append(f"  Document changed: {changed}")

    return tex, "\n".join(report_lines)


def _handle_font_setup(tex: str, font_name: str) -> str:
    """Handle XeLaTeX font setup."""
    # Replace tabs with spaces
    tex = tex.replace("\t", "  ")

    # Fix font setup if present
    if re.search(r"\\setmainfont(?:\[[^\]]*\])?\{[^}]+\}", tex):
        tex = re.sub(
            r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
            rf"\1{font_name}\3",
            tex,
        )
    else:
        # Add font setup after fontspec if missing
        m = re.search(r"(\\usepackage(?:\[[^\]]*\])?\{fontspec\}\s*)", tex)
        if m:
            insert_at = m.end()
            tex = tex[:insert_at] + f"\\setmainfont{{{font_name}}}\n" + tex[insert_at:]

    return tex


# For backwards compatibility
def clean_tex(tex: str) -> Tuple[str, Dict]:
    """Backwards compatible interface."""
    cleaned, report_str = clean_tex_robust(tex)

    # Parse fixes from report for dict format
    fixes = {}
    for line in report_str.split('\n'):
        if ':' in line and 'fixed' in line or 'converted' in line or 'sanitized' in line:
            parts = line.strip().split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    value = int(parts[1].strip())
                    fixes[key] = value
                except ValueError:
                    pass

    return cleaned, fixes