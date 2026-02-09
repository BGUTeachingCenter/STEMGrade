"""
Extremely robust LaTeX cleaning for XeLaTeX compilation.
Handles all common issues with AI-generated LaTeX content.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import re

_LEFT_RE = re.compile(r"\\left\s*")
_RIGHT_RE = re.compile(r"\\right\s*")

def _fix_unmatched_left_right(math: str) -> Tuple[str, int]:
    """
    Fix LaTeX errors: 'Extra \\right.' or 'Extra \\left.'
    Strategy:
      - Token-scan for \\left and \\right
      - If \\right appears with no open \\left, drop the \\right (keep the delimiter char)
      - If \\left remains unmatched at the end, drop those \\left tokens (keep delimiter char)
    This preserves delimiters like '{' '}' '(' ')' while removing the fragile sizing commands.
    """
    if not math:
        return math, 0

    # Find all \left and \right occurrences
    # We'll scan left-to-right and build output while tracking unmatched \left tokens.
    fixes = 0
    out = []
    i = 0
    n = len(math)

    stack = 0  # count of open \left not yet matched by \right

    while i < n:
        mL = _LEFT_RE.match(math, i)
        if mL:
            # Tentatively include \left, but we may drop unmatched later.
            out.append(mL.group(0))
            stack += 1
            i = mL.end()
            continue

        mR = _RIGHT_RE.match(math, i)
        if mR:
            if stack <= 0:
                # Unmatched \right -> drop it
                fixes += 1
                # do not append anything; the delimiter char follows and will remain
            else:
                out.append(mR.group(0))
                stack -= 1
            i = mR.end()
            continue

        out.append(math[i])
        i += 1

    fixed = "".join(out)

    # If there are still unmatched \left tokens, remove that many from the left-to-right stream.
    # Easiest: remove ALL \left tokens if no \right exists; otherwise remove extra \left from the end.
    if stack > 0:
        # remove the last `stack` occurrences of \left (keep delimiters)
        parts = list(_LEFT_RE.finditer(fixed))
        if parts:
            fixes += stack
            # remove from the end
            to_remove = parts[-stack:]
            mask = [True] * len(fixed)
            for m in to_remove:
                for j in range(m.start(), m.end()):
                    mask[j] = False
            fixed = "".join(ch for ch, keep in zip(fixed, mask) if keep)

    return fixed, fixes


_ESCAPED_DOLLAR = re.compile(r"\\\$")
_MATH_ENV_NAMES = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "flalign", "flalign*", "alignat", "alignat*",
}

_BEGIN_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_END_ENV_RE   = re.compile(r"\\end\{([a-zA-Z*]+)\}")

def _looks_like_display(math_text: str) -> bool:
    """Heuristic: decide if content is display-ish."""
    t = math_text.strip()
    if not t:
        return False
    # Any explicit line breaks / alignment / big operators tends to be display
    if r"\\" in t or "&" in t:
        return True
    if any(cmd in t for cmd in (r"\sum", r"\int", r"\prod", r"\lim", r"\frac", r"\begin")):
        return True
    # Long-ish expressions are safer as display
    if len(t) > 45:
        return True
    return False

def _balance_math_delimiters(
    tex: str,
    *,
    prefer_paren_bracket: bool = True,   # convert $..$ -> \(...\), $$..$$ -> \[...\]
    force_inline_on_mismatch: bool = True # when $$...$ happens, treat as inline
) -> Tuple[str, int]:
    """
    Walk the document and normalize $ / $$ delimiters with a small state machine.
    Fixes $$...$ and $...$$ mismatches + unclosed math.
    Returns (new_tex, fixes_count).
    """
    if not tex:
        return tex, 0

    i = 0
    n = len(tex)
    out = []
    fixes = 0

    # states: "text", "inline", "display"
    state = "text"
    opener = None  # "$" or "$$"
    math_buf = []  # collect math content when in math state

    # Track whether we are inside environments where $ should be left alone
    # (verbatim-ish). Extend if you use minted/listings.
    verbatim_env_stack = []
    VERBATIM_ENVS = {"verbatim", "Verbatim", "lstlisting", "minted"}

    def flush_math(close_as: str):
        nonlocal fixes
        content = "".join(math_buf)
        math_buf.clear()

        # Optionally convert to \(...\) / \[...\] for robustness
        if prefer_paren_bracket:
            if close_as == "$":
                out.append(r"\(" + content + r"\)")
            else:
                out.append(r"\[" + content + r"\]")
        else:
            out.append(close_as + content + close_as)

    while i < n:
        ch = tex[i]

        # --- environment tracking (very lightweight) ---
        if state == "text":
            m = _BEGIN_ENV_RE.match(tex, i)
            if m:
                env = m.group(1)
                out.append(m.group(0))
                i += len(m.group(0))
                if env in VERBATIM_ENVS:
                    verbatim_env_stack.append(env)
                continue
            m = _END_ENV_RE.match(tex, i)
            if m:
                env = m.group(1)
                out.append(m.group(0))
                i += len(m.group(0))
                if verbatim_env_stack and verbatim_env_stack[-1] == env:
                    verbatim_env_stack.pop()
                continue

        # If we're in verbatim-like env, do not touch dollars at all
        if verbatim_env_stack:
            out.append(ch)
            i += 1
            continue

        # Handle escaped dollar
        if tex.startswith(r"\$", i):
            if state == "text":
                out.append(r"\$")
            else:
                math_buf.append(r"\$")
            i += 2
            continue

        # Detect $$ or $
        if ch == "$":
            is_double = (i + 1 < n and tex[i + 1] == "$")
            token = "$$" if is_double else "$"

            if state == "text":
                # open math
                state = "display" if is_double else "inline"
                opener = token
                if not prefer_paren_bracket:
                    out.append(token)
                i += 2 if is_double else 1
                continue

            # If we are in inline math and see $$, this is likely corruption: $ ... $$
            if state == "inline" and token == "$$":
                # Close inline math first
                flush_math("$")
                fixes += 1
                state = "text"
                opener = None
                i += 2
                continue

            # If we are in display math and see single $, this is likely corruption: $$ ... $
            if state == "display" and token == "$":
                content = "".join(math_buf)
                if force_inline_on_mismatch and not _looks_like_display(content):
                    # Treat it as inline: $$ content $  -> inline
                    flush_math("$")
                else:
                    # Upgrade closing $ to $$ (i.e., keep display)
                    flush_math("$$")
                fixes += 1
                state = "text"
                opener = None
                i += 1
                continue

            # Normal closing
            if state == "inline" and token == "$":
                flush_math("$")
                state = "text"
                opener = None
                i += 1
                continue

            if state == "display" and token == "$$":
                flush_math("$$")
                state = "text"
                opener = None
                i += 2
                continue

            # Any other weird combo: force-close in the safest way
            if state in ("inline", "display"):
                close_as = "$" if state == "inline" else "$$"
                flush_math(close_as)
                fixes += 1
                state = "text"
                opener = None
                i += 2 if is_double else 1
                continue

        # Regular character
        if state == "text":
            out.append(ch)
        else:
            math_buf.append(ch)
        i += 1

    # If file ends while math is still open, close it
    if state in ("inline", "display"):
        close_as = "$" if state == "inline" else "$$"
        flush_math(close_as)
        fixes += 1

    return "".join(out), fixes



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

_ITEMIZE_BLOCK_RE = re.compile(
    r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
    re.DOTALL | re.IGNORECASE
)

def _fix_itemize_blocks(text: str) -> Tuple[str, int]:
    """
    Make itemize environments compile-safe:
    - Convert bullet-ish lines ("-", "*", "•") into \item
    - If an itemize block has no \item at all, wrap the whole content as one \item
    - Remove empty itemize blocks
    """
    fixes = 0

    def _repair_block(m: re.Match) -> str:
        nonlocal fixes
        inner = m.group(1)

        # Normalize line endings inside block
        lines = inner.splitlines()

        new_lines: List[str] = []
        saw_item = False
        for line in lines:
            raw = line.rstrip()

            # Keep blank lines, but don't let them be the only content
            if not raw.strip():
                new_lines.append(raw)
                continue

            s = raw.lstrip()

            # Already a proper item
            if s.startswith(r"\item"):
                saw_item = True
                new_lines.append(raw)
                continue

            # Common LLM bullet formats -> \item
            if s.startswith(("-", "*", "•", "–", "—")):
                # remove the bullet marker
                content = s[1:].lstrip()
                saw_item = True
                new_lines.append(r"\item " + content)
                fixes += 1
                continue

            # If line begins with \textbf{...}: treat as item content if we are in list
            # (many models output headings without \item)
            # We'll only convert to \item if we haven't seen any item yet OR previous nonblank was an \item
            # Safer rule: if no \item exists in whole block, we'll wrap later.
            new_lines.append(raw)

        repaired_inner = "\n".join(new_lines).strip()

        # Remove completely empty lists
        if not repaired_inner:
            fixes += 1
            return ""  # delete block

        # If still no \item anywhere, wrap entire content as one item
        if r"\item" not in repaired_inner:
            fixes += 1
            return "\\begin{itemize}\n\\item " + repaired_inner + "\n\\end{itemize}"

        return "\\begin{itemize}\n" + repaired_inner + "\n\\end{itemize}"

    new_text, n = _ITEMIZE_BLOCK_RE.subn(_repair_block, text)
    fixes += n  # counts blocks touched (roughly)
    return new_text, fixes


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

    # Fix unmatched \left/\right that would crash XeLaTeX
    content, _n_lr = _fix_unmatched_left_right(content)


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

    tex, n = _balance_math_delimiters(tex, prefer_paren_bracket=True)
    if n: fixes["math_delimiters_balanced"] = n

    # Phase 4: Content cleaning
    tex, n = _fix_undefined_commands(tex)
    if n: fixes["undefined_commands_fixed"] = n

    tex, n = _fix_invalid_delimiters(tex)
    if n: fixes["invalid_delimiters_fixed"] = n

    tex, n = _fix_itemize_blocks(tex)
    if n: fixes["itemize_blocks_fixed"] = n

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