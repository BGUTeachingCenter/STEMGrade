"""
Extremely robust LaTeX cleaning for XeLaTeX compilation.
Handles common issues with AI-generated LaTeX and mixed Hebrew/math.
All cleaning lives here. The compiler should ONLY compile.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .ollama_proofreader import proofread_tex_with_ollama


# ============================================================
# Helpers: safe brace and delimiter repairs inside MATH only
# ============================================================

# Wrap patterns like: ב-\sup{ ... } / ב-\inf{ ... } that appear in TEXT mode
# and also fix common plain-text tokens mid/in -> \mid/\in inside that wrapped chunk.
_OP_TEXT_RE = re.compile(
    r"""
    (?P<prefix>ב-)\s*-\s*\\(?P<op>sup|inf|max|min|lim)\{   # "ב-\sup{"
    (?P<body>.*?)
    (?P<closer>\\\}|})                                   # either "\}" or "}"
    """,
    re.VERBOSE | re.DOTALL,
)

def _wrap_operator_phrase_math_in_text(tex: str) -> tuple[str, int]:
    """
    Fix the very common grading-output pattern:
      ... ב-\sup{ ... } ...
    when it appears in TEXT MODE (outside any existing \( \) / \[ \] / $ $).

    We wrap it as:
      ... ב-\(\sup\{ ... \}\) ...

    And we normalize inside the wrapped math:
      - ' mid ' -> ' \\mid '
      - ' in '  -> ' \\in '
      - '\\_'   -> '_'  (underscores should not be escaped inside math)
      - '\\f('  -> 'f(' (common hallucination/typo)
    """
    if not tex:
        return tex, 0

    fixes = 0
    out = []
    i = 0
    n = len(tex)
    state = "text"  # text|inline|display

    def starts(s: str) -> bool:
        return tex.startswith(s, i)

    while i < n:
        # --- math boundaries ---
        if state == "text" and starts(r"\("):
            state = "inline"; out.append(r"\("); i += 2; continue
        if state == "inline" and starts(r"\)"):
            state = "text"; out.append(r"\)"); i += 2; continue
        if state == "text" and starts(r"\["):
            state = "display"; out.append(r"\["); i += 2; continue
        if state == "display" and starts(r"\]"):
            state = "text"; out.append(r"\]"); i += 2; continue

        # $ / $$ (rare, but handle)
        if tex[i] == "$":
            dbl = (i + 1 < n and tex[i + 1] == "$")
            if state == "text":
                state = "display" if dbl else "inline"
            else:
                if state == "inline" and not dbl: state = "text"
                elif state == "display" and dbl: state = "text"
            out.append("$$" if dbl else "$")
            i += 2 if dbl else 1
            continue

        if state != "text":
            out.append(tex[i])
            i += 1
            continue

        # --- TEXT MODE: try to match the operator phrase from this position ---
        m = _OP_TEXT_RE.match(tex, i)
        if not m:
            out.append(tex[i]); i += 1; continue

        prefix = m.group("prefix")
        op = m.group("op")
        body = m.group("body")

        # normalize within the math body
        body = body.replace(r"\_", "_")
        body = body.replace(r"\f(", "f(")
        body = re.sub(r"(?<!\\)\bmid\b", r"\\mid", body)  # mid -> \mid
        body = re.sub(r"(?<!\\)\bin\b", r"\\in", body)    # in  -> \in

        # ensure set braces are explicit \{ \}
        replacement = (
            f"{prefix}-\\(\\{op}\\{{{body}\\}}\\)"
        )

        out.append(replacement)
        fixes += 1
        i = m.end()

    return "".join(out), fixes

def _close_unbalanced_inline_paren_math(tex: str) -> tuple[str, int]:
    """
    Global fix: if there are more '\\(' than '\\)', append missing '\\)' at end.
    This prevents math-mode leaking to later environments.
    """
    if not tex:
        return tex, 0
    opens = tex.count(r"\(")
    closes = tex.count(r"\)")
    if opens > closes:
        diff = opens - closes
        return tex + ("\n" + (r"\)" * diff) + "\n"), diff
    return tex, 0

def _fix_alignment_tokens_in_math(tex: str) -> tuple[str, int]:
    """
    Remove/neutralize alignment tokens inside math blocks that commonly appear
    from AI/student LaTeX snippets:
      - '&' inside \\(...\\) or \\[...\\] -> replace with ',' (safe separator)
      - '\\\\' inside \\(...\\) -> replace with ';\\;' (safe separator)
    """
    if not tex:
        return tex, 0

    fixes = 0

    # Inline: \( ... \)
    rx_inline = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

    def repl_inline(m: re.Match) -> str:
        nonlocal fixes
        inner = m.group(1)
        if "&" in inner:
            inner2, n = re.subn(r"(?<!\\)&", ",", inner)  # don't touch \&
            fixes += n
            inner = inner2
        if r"\\" in inner:
            inner2, n = re.subn(r"\\\\", r";\,", inner)   # newline -> separator
            fixes += n
            inner = inner2
        return r"\(" + inner + r"\)"

    tex = rx_inline.sub(repl_inline, tex)

    # Display: \[ ... \]
    rx_disp = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

    def repl_disp(m: re.Match) -> str:
        nonlocal fixes
        inner = m.group(1)
        if "&" in inner:
            inner2, n = re.subn(r"(?<!\\)&", ",", inner)
            fixes += n
            inner = inner2
        # display math may legitimately contain \\ but if it's not in aligned,
        # we still prefer to soften it
        if r"\\" in inner and r"\begin" not in inner:
            inner2, n = re.subn(r"\\\\", r"\\,", inner)
            fixes += n
            inner = inner2
        return r"\[" + inner + r"\]"

    tex = rx_disp.sub(repl_disp, tex)

    return tex, fixes


def _close_unbalanced_inline_paren_math_global(tex: str) -> tuple[str, int]:
    """
    If the document has more \\( than \\), append missing \\) at the end.
    Conservative: does NOT try to guess where to close—just prevents leaking math mode.
    """
    if not tex:
        return tex, 0
    opens = len(re.findall(r"\\\(", tex))
    closes = len(re.findall(r"\\\)", tex))
    if opens > closes:
        missing = opens - closes
        return tex + ("\n" + (r"\)" * missing) + "\n"), missing
    return tex, 0


def _balance_paren_bracket_math_globally(tex: str) -> tuple[str, int]:
    """
    Global balancing for \\( \\) and \\[ \\].
    This prevents 'Bad math environment delimiter' without the dangerous per-line approach.
    Strategy:
      - If opens > closes, append missing closes at end of document.
      - If closes > opens, remove extra closing tokens from the end (safest).
    """
    if not tex:
        return tex, 0

    fixes = 0

    # --- balance \( \) ---
    opens = len(re.findall(r"\\\(", tex))
    closes = len(re.findall(r"\\\)", tex))

    if opens > closes:
        diff = opens - closes
        tex = tex + ("\n" + (r"\)" * diff) + "\n")
        fixes += diff
    elif closes > opens:
        diff = closes - opens
        for _ in range(diff):
            # remove last occurrence of \)
            tex2 = re.sub(r"(.*)\\\)(?!.*\\\))", r"\1", tex, flags=re.DOTALL)
            if tex2 == tex:
                break
            tex = tex2
            fixes += 1

    # --- balance \[ \] ---
    opens = len(re.findall(r"\\\[", tex))
    closes = len(re.findall(r"\\\]", tex))

    if opens > closes:
        diff = opens - closes
        tex = tex + ("\n" + (r"\]" * diff) + "\n")
        fixes += diff
    elif closes > opens:
        diff = closes - opens
        for _ in range(diff):
            tex2 = re.sub(r"(.*)\\\](?!.*\\\])", r"\1", tex, flags=re.DOTALL)
            if tex2 == tex:
                break
            tex = tex2
            fixes += 1

    return tex, fixes

def _fix_operator_set_braces_in_math(tex: str) -> tuple[str, int]:
    """
    Fix common model output bug inside math:
      \\sup{ ... \\}   -> \\sup\\{ ... \\}
      \\inf{ ... \\}   -> \\inf\\{ ... \\}
      \\max{ ... \\}   -> \\max\\{ ... \\}
      \\min{ ... \\}   -> \\min\\{ ... \\}
      \\lim{ ... \\}   -> \\lim\\{ ... \\}

    We only apply when the operator is followed by '{' AND there is a literal '\\}'
    later in the SAME math block (meaning the intent was set braces).

    IMPORTANT: We must allow '}' inside the block (e.g. x_{k-1}), so we use a
    permissive lookahead (?=.*?\\}) instead of [^}]*.
    """
    if not tex:
        return tex, 0

    fixes = 0

    # Work on inline math blocks: \( ... \)
    rx_inline = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

    def fix_block(block: str) -> str:
        nonlocal fixes
        for op in ("sup", "inf", "max", "min", "lim"):
            # match \op{ ... } only if somewhere ahead there is a literal '\}'
            pat = re.compile(rf"\\{op}\{{(?=.*?\\\}})", re.DOTALL)
            block, n = pat.subn(rf"\\{op}\\{{", block)
            fixes += n
        return block

    def repl_inline(m: re.Match) -> str:
        inner = m.group(1)
        return r"\(" + fix_block(inner) + r"\)"

    tex = rx_inline.sub(repl_inline, tex)

    # display math blocks: \[ ... \]
    rx_disp = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

    def repl_disp(m: re.Match) -> str:
        inner = m.group(1)
        return r"\[" + fix_block(inner) + r"\]"

    tex = rx_disp.sub(repl_disp, tex)

    # $$ ... $$ blocks
    rx_dollars = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

    def repl_dollars(m: re.Match) -> str:
        inner = m.group(1)
        return "$$" + fix_block(inner) + "$$"

    tex = rx_dollars.sub(repl_dollars, tex)

    return tex, fixes

# def _close_unbalanced_inline_paren_math_per_line(tex: str) -> tuple[str, int]:
#     """
#     Fix lines that contain '\\(' but not '\\)' by appending '\\)' to the SAME LINE.
#     This prevents leaking math mode into environments (e.g., itemize).
#     """
#     if not tex:
#         return tex, 0
#
#     fixes = 0
#     out_lines: list[str] = []
#
#     for line in tex.splitlines():
#         if r"\(" in line and r"\)" not in line:
#             # Don't touch lines that already have an explicit environment boundary
#             # (but item lines are safe to close).
#             out_lines.append(line + r"\)")
#             fixes += 1
#         else:
#             out_lines.append(line)
#
#     return "\n".join(out_lines), fixes


def _wrap_mathy_item_lines(tex: str) -> tuple[str, int]:
    """
    Wrap \\item lines that contain obvious LaTeX math commands/fragments
    into \\( ... \\) if they are not already delimited.

    This prevents 'Missing $ inserted' and keeps _/^ legal.
    """
    if not tex:
        return tex, 0

    # Heuristic: looks like math if it contains key commands OR _/^ patterns OR '=' with digits
    mathy_re = re.compile(
        r"(\\(int|sum|prod|frac|sqrt|lim|cdot|to|infty|Delta|delta|epsilon|lambda|pi|theta|alpha|beta|gamma"
        r"|sin|cos|tan|ln|log|exp)\b|"
        r"[_^]\{?[\w\d]|"
        r"\d\s*[=<>]\s*\d|"
        r"\\leq|\\geq|\\neq)"
    )

    item_re = re.compile(r"^(?P<pre>\s*\\item\s+)(?P<body>.+)$", re.MULTILINE)

    def repl(m: re.Match) -> str:
        pre = m.group("pre")
        body = m.group("body").rstrip()

        # If already has explicit math delimiters, do nothing.
        if (r"\(" in body) or (r"\[" in body) or ("$" in body):
            return m.group(0)

        # Only wrap if it looks mathy
        if not mathy_re.search(body):
            return m.group(0)

        return pre + r"\(" + body + r"\)"

    new_tex, n = item_re.subn(repl, tex)
    return new_tex, n


def _wrap_item_math_lines(tex: str) -> tuple[str, int]:
    """
    If an item line begins with a common math command (\\int, \\sum, \\frac, etc.)
    in TEXT MODE, wrap the rest of the line in \\( ... \\).

    This prevents: "Missing $ inserted" and avoids escaping _/^ as text.
    We only touch lines that:
      - start with \\item
      - contain one of the math commands early
      - do NOT already contain \\(, \\[, or $ on that line
    """
    if not tex:
        return tex, 0

    math_cmd = r"(?:\\int|\\sum|\\prod|\\lim|\\frac|\\sqrt|\\log|\\ln|\\sin|\\cos|\\tan)"
    rx = re.compile(
        rf"^(?P<prefix>\s*\\item\s+)(?P<body>{math_cmd}\b.*)$",
        flags=re.MULTILINE
    )

    def repl(m: re.Match) -> str:
        prefix = m.group("prefix")
        body = m.group("body")

        # If the line already has explicit math delimiters, do nothing.
        if (r"\(" in body) or (r"\[" in body) or ("$" in body):
            return m.group(0)

        # Wrap the whole remainder of the item line in inline math
        return prefix + r"\(" + body.rstrip() + r"\)"

    new_tex, n = rx.subn(repl, tex)
    return new_tex, n


def _final_fix_double_backslash_underscore_in_text(tex: str) -> tuple[str, int]:
    """
    FINAL safety pass:
    In TEXT MODE only, collapse any '\\\\_' (newline + underscore) into '\\_'.
    Also collapse '\\\\\\_' (newline + escaped underscore) into '\\_'.
    We do NOT touch math mode.

    This prevents: ! Missing $ inserted ... algebra\\\\_error
    """
    if not tex:
        return tex, 0

    out: list[str] = []
    fixes = 0
    i = 0
    n = len(tex)

    state = "text"  # text|inline|display

    def starts(s: str) -> bool:
        return tex.startswith(s, i)

    while i < n:
        # --- math mode boundaries ---
        if state == "text" and starts(r"\("):
            state = "inline"; out.append(r"\("); i += 2; continue
        if state == "inline" and starts(r"\)"):
            state = "text"; out.append(r"\)"); i += 2; continue
        if state == "text" and starts(r"\["):
            state = "display"; out.append(r"\["); i += 2; continue
        if state == "display" and starts(r"\]"):
            state = "text"; out.append(r"\]"); i += 2; continue

        # $ / $$ (just in case any remain)
        if tex[i] == "$":
            dbl = (i + 1 < n and tex[i + 1] == "$")
            if state == "text":
                state = "display" if dbl else "inline"
            else:
                if state == "inline" and not dbl: state = "text"
                elif state == "display" and dbl: state = "text"
            out.append("$$" if dbl else "$")
            i += 2 if dbl else 1
            continue

        # --- the critical fix: TEXT MODE only ---
        # literal "\\_"  -> "\_"
        if state == "text" and starts(r"\\_"):
            out.append(r"\_")
            fixes += 1
            i += 3
            continue

        # literal "\\\_" -> "\_"  (newline + already-escaped underscore)
        if state == "text" and starts(r"\\\_"):
            out.append(r"\_")
            fixes += 1
            i += 4
            continue

        out.append(tex[i])
        i += 1

    return "".join(out), fixes

def _fix_newline_underscore_in_text_mode(tex: str) -> tuple[str, int]:
    """
    Fix the very common LLM bug: '\\\\_' (newline + underscore) in TEXT MODE.
    Convert it to '\\_'.

    We do this with a small state machine so we do NOT touch math mode.
    Math modes recognized:
      - \\(...\\)
      - \\[...\\]
      - $...$
      - $$...$$
    """
    if not tex:
        return tex, 0

    out: list[str] = []
    fixes = 0
    i = 0
    n = len(tex)

    state = "text"  # text|inline|display

    while i < n:
        # --- enter/exit LaTeX math via \( \) and \[ \] ---
        if state == "text" and tex.startswith(r"\(", i):
            state = "inline"
            out.append(r"\(")
            i += 2
            continue
        if state == "inline" and tex.startswith(r"\)", i):
            state = "text"
            out.append(r"\)")
            i += 2
            continue
        if state == "text" and tex.startswith(r"\[", i):
            state = "display"
            out.append(r"\[")
            i += 2
            continue
        if state == "display" and tex.startswith(r"\]", i):
            state = "text"
            out.append(r"\]")
            i += 2
            continue

        # --- enter/exit math via $ / $$ (in case any remain) ---
        if tex[i] == "$":
            dbl = (i + 1 < n and tex[i + 1] == "$")
            if state == "text":
                state = "display" if dbl else "inline"
            else:
                if state == "inline" and not dbl:
                    state = "text"
                elif state == "display" and dbl:
                    state = "text"
            out.append("$$" if dbl else "$")
            i += 2 if dbl else 1
            continue

        # --- the actual fix: in TEXT MODE only, convert \\_ -> \_ ---
        if state == "text" and tex.startswith(r"\\_", i):
            out.append(r"\_")
            fixes += 1
            i += 3
            continue

        # also fix the (rarer) case of \\^ in text mode
        if state == "text" and tex.startswith(r"\\^", i):
            out.append(r"\textasciicircum{}")
            fixes += 1
            i += 3
            continue

        out.append(tex[i])
        i += 1

    return "".join(out), fixes


def _fix_newline_escape_before_underscore(tex: str) -> tuple[str, int]:
    """
    Fix the very common LLM bug: writing '\\\\_' in text mode.
    In LaTeX, '\\\\' is a newline command; '\\\\_' becomes 'newline + _' which breaks compilation.
    Convert '\\\\_' -> '\\_'.

    We ONLY fix the exact sequence '\\\\_' (two backslashes + underscore).
    """
    if not tex:
        return tex, 0
    # Match two backslashes not preceded by another backslash to avoid eating '\\\\\_'
    # i.e. replace occurrences of \\_ (newline underscore) with \_ (escaped underscore)
    new_tex, n = re.subn(r"(?<!\\)\\\\_", r"\\_", tex)
    return new_tex, n

def _fix_newline_escape_before_caret(tex: str) -> tuple[str, int]:
    """
    Convert '\\\\^' (newline + caret) into a safe caret in text mode.
    """
    if not tex:
        return tex, 0
    new_tex, n = re.subn(r"(?<!\\)\\\\\^", r"\\textasciicircum{}", tex)
    return new_tex, n

def _balance_curly_braces_in_math(math: str) -> tuple[str, int]:
    """
    Balance { } inside math blocks only.
    Conservative: ignores escaped \{ \}.
    If opens > closes, append missing '}'.
    If closes > opens, remove extra '}' from the end.
    """
    if not math:
        return math, 0

    opens = closes = 0
    i = 0
    n = len(math)
    while i < n:
        if math.startswith(r"\{", i) or math.startswith(r"\}", i):
            i += 2
            continue
        ch = math[i]
        if ch == "{":
            opens += 1
        elif ch == "}":
            closes += 1
        i += 1

    diff = opens - closes
    if diff > 0:
        return math + ("}" * diff), diff
    if diff < 0:
        remove = -diff
        out = list(math)
        j = len(out) - 1
        while j >= 0 and remove > 0:
            if out[j] == "}":
                out.pop(j)
                remove -= 1
            j -= 1
        return "".join(out), -diff
    return math, 0


_LEFT_RE = re.compile(r"\\left\s*")
_RIGHT_RE = re.compile(r"\\right\s*")

def _fix_unmatched_left_right(math: str) -> Tuple[str, int]:
    """
    Fix 'Extra \\right.' / unmatched \\left.
    Drops unmatched \\right tokens and removes extra \\left tokens (keeping delimiters).
    """
    if not math:
        return math, 0

    fixes = 0
    out: list[str] = []
    i = 0
    n = len(math)
    stack = 0

    while i < n:
        mL = _LEFT_RE.match(math, i)
        if mL:
            out.append(mL.group(0))
            stack += 1
            i = mL.end()
            continue

        mR = _RIGHT_RE.match(math, i)
        if mR:
            if stack <= 0:
                fixes += 1
                # drop \right (delimiter char remains)
            else:
                out.append(mR.group(0))
                stack -= 1
            i = mR.end()
            continue

        out.append(math[i])
        i += 1

    fixed = "".join(out)

    if stack > 0:
        parts = list(_LEFT_RE.finditer(fixed))
        if parts:
            fixes += stack
            to_remove = parts[-stack:]
            mask = [True] * len(fixed)
            for m in to_remove:
                for j in range(m.start(), m.end()):
                    mask[j] = False
            fixed = "".join(ch for ch, keep in zip(fixed, mask) if keep)

    return fixed, fixes


# ============================================================
# Dollar delimiter balancing (document-level, but purely structural)
# ============================================================

_BEGIN_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_END_ENV_RE   = re.compile(r"\\end\{([a-zA-Z*]+)\}")

def _looks_like_display(math_text: str) -> bool:
    t = math_text.strip()
    if not t:
        return False
    if r"\\" in t or "&" in t:
        return True
    if any(cmd in t for cmd in (r"\sum", r"\int", r"\prod", r"\lim", r"\frac", r"\begin")):
        return True
    return len(t) > 45


def _balance_math_delimiters(
    tex: str,
    *,
    prefer_paren_bracket: bool = True,   # $..$ -> \(...\), $$..$$ -> \[...\]
    force_inline_on_mismatch: bool = True
) -> Tuple[str, int]:
    """
    Normalize $/$$ delimiters with a small state machine.
    Optionally converts to \(..\) and \[..\] for robustness.
    """
    if not tex:
        return tex, 0

    i = 0
    n = len(tex)
    out: list[str] = []
    fixes = 0

    state = "text"   # text|inline|display
    math_buf: list[str] = []

    verbatim_env_stack: list[str] = []
    VERBATIM_ENVS = {"verbatim", "Verbatim", "lstlisting", "minted"}

    def flush_math(close_as: str) -> None:
        content = "".join(math_buf)
        math_buf.clear()

        if prefer_paren_bracket:
            if close_as == "$":
                out.append(r"\(" + content + r"\)")
            else:
                out.append(r"\[" + content + r"\]")
        else:
            out.append(close_as + content + close_as)

    while i < n:
        ch = tex[i]

        # env tracking in text
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

        if verbatim_env_stack:
            out.append(ch)
            i += 1
            continue

        # escaped dollar
        if tex.startswith(r"\$", i):
            if state == "text":
                out.append(r"\$")
            else:
                math_buf.append(r"\$")
            i += 2
            continue

        if ch == "$":
            is_double = (i + 1 < n and tex[i + 1] == "$")
            token = "$$" if is_double else "$"

            if state == "text":
                state = "display" if is_double else "inline"
                i += 2 if is_double else 1
                continue

            if state == "inline" and token == "$$":
                flush_math("$")
                fixes += 1
                state = "text"
                i += 2
                continue

            if state == "display" and token == "$":
                content = "".join(math_buf)
                if force_inline_on_mismatch and not _looks_like_display(content):
                    flush_math("$")
                else:
                    flush_math("$$")
                fixes += 1
                state = "text"
                i += 1
                continue

            if state == "inline" and token == "$":
                flush_math("$")
                state = "text"
                i += 1
                continue

            if state == "display" and token == "$$":
                flush_math("$$")
                state = "text"
                i += 2
                continue

            # fallback
            flush_math("$" if state == "inline" else "$$")
            fixes += 1
            state = "text"
            i += 2 if is_double else 1
            continue

        # regular char
        if state == "text":
            out.append(ch)
        else:
            math_buf.append(ch)
        i += 1

    if state in ("inline", "display"):
        flush_math("$" if state == "inline" else "$$")
        fixes += 1

    return "".join(out), fixes


# ============================================================
# Regex patterns for math blocks we sanitize
# ============================================================

_MATH_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE_RE  = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)
_BRACKET_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_PAREN_INLINE_RE    = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

_MATH_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|alignat\*?)\}"
    r"(.*?)"
    r"\\end\{\1\}",
    re.DOTALL | re.IGNORECASE
)

_MULTI_DOLLAR_RE = re.compile(r"(?<!\\)\${3,}")
_TEXTBACKSLASH_RE = re.compile(r"\\textbackslash\{\}")
_TEXTBRACE_RE = re.compile(r"\\textbraceleft|\\textbraceright")

_INVALID_LEFT_BRACE_RE  = re.compile(r"\\left\s*\{", re.IGNORECASE)
_INVALID_RIGHT_BRACE_RE = re.compile(r"\\right\s*\}", re.IGNORECASE)

_ITEMIZE_BLOCK_RE = re.compile(
    r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
    re.DOTALL | re.IGNORECASE
)

_UNDEFINED_COMMANDS = {
    r'\\abs\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"|{m.group(1)}|",
    r'\\norm\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\|{m.group(1)}\\|",
    r'\\floor\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\lfloor {m.group(1)} \\rfloor",
    r'\\ceil\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': lambda m: f"\\lceil {m.group(1)} \\rceil",
}

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


# ============================================================
# Small targeted repairs
# ============================================================

def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _fix_textbackslash_sequences(text: str) -> Tuple[str, int]:
    count = 0
    text, n1 = _TEXTBACKSLASH_RE.subn("\\\\", text)
    count += n1
    text, n2 = _TEXTBRACE_RE.subn("", text)
    count += n2
    return text, count


def _fix_multi_dollars(text: str) -> Tuple[str, int]:
    return _MULTI_DOLLAR_RE.subn("$$", text)


def _fix_double_escaped_underscore(tex: str) -> tuple[str, int]:
    # turns \\_ into \_
    return re.subn(r"\\\\_", r"\\_", tex)


def _fix_escaped_dollars_in_commands(text: str) -> Tuple[str, int]:
    patterns = [
        (r"\\\$\\begin\{", r"\\begin{"),
        (r"\\\$\\end\{", r"\\end{"),
        (r"\\\$\s*(\\[a-zA-Z]+)", r"\1"),
    ]
    count = 0
    for pattern, replacement in patterns:
        text, n = re.subn(pattern, replacement, text)
        count += n
    return text, count


def _fix_malformed_environments(text: str) -> Tuple[str, int]:
    fixes = [
        (r"\\begin\\?\{([^}]+)\\?\}", r"\\begin{\1}"),
        (r"\\end\\?\{([^}]+)\\?\}", r"\\end{\1}"),
        (r"\\\\([a-zA-Z]+)", r"\\\1"),
        (r"\\([a-zA-Z]+)\\?\{", r"\\\1{"),
        (r"\\newline\s*\\item", r"\\item"),
        (r"\\par\s*\\item", r"\\item"),
    ]
    count = 0
    for pattern, replacement in fixes:
        text, n = re.subn(pattern, replacement, text)
        count += n
    return text, count


def _fix_invalid_delimiters(text: str) -> Tuple[str, int]:
    count = 0
    text, n1 = _INVALID_LEFT_BRACE_RE.subn(r"\\{", text)
    count += n1
    text, n2 = _INVALID_RIGHT_BRACE_RE.subn(r"\\}", text)
    count += n2
    return text, count


def _fix_undefined_commands(text: str) -> Tuple[str, int]:
    count = 0
    for pattern, repl in _UNDEFINED_COMMANDS.items():
        text, n = re.subn(pattern, repl, text)
        count += n
    return text, count


def _fix_itemize_blocks(text: str) -> Tuple[str, int]:
    fixes = 0

    def _repair_block(m: re.Match) -> str:
        nonlocal fixes
        inner = m.group(1)
        lines = inner.splitlines()

        new_lines: List[str] = []
        for line in lines:
            raw = line.rstrip()
            if not raw.strip():
                new_lines.append(raw)
                continue

            s = raw.lstrip()
            if s.startswith(r"\item"):
                new_lines.append(raw)
                continue

            if s.startswith(("-", "*", "•", "–", "—")):
                content = s[1:].lstrip()
                new_lines.append(r"\item " + content)
                fixes += 1
                continue

            new_lines.append(raw)

        repaired_inner = "\n".join(new_lines).strip()
        if not repaired_inner:
            fixes += 1
            return ""

        if r"\item" not in repaired_inner:
            fixes += 1
            return "\\begin{itemize}\n\\item " + repaired_inner + "\n\\end{itemize}"

        return "\\begin{itemize}\n" + repaired_inner + "\n\\end{itemize}"

    new_text, _nblocks = _ITEMIZE_BLOCK_RE.subn(_repair_block, text)
    return new_text, fixes


# ============================================================
# Math sanitization (critical)
# ============================================================

def _clean_math_content(math_content: str) -> str:
    if not math_content:
        return math_content

    content = math_content

    content, _ = _fix_undefined_commands(content)

    for escaped, unescaped in _MATH_UNESCAPE_MAP.items():
        content = content.replace(escaped, unescaped)

    content, _ = _fix_invalid_delimiters(content)

    # prevent % comments inside math (common)
    content = content.replace("%", r"\%")

    content, _ = _fix_unmatched_left_right(content)

    # balance braces inside math ONLY
    content, _ = _balance_curly_braces_in_math(content)

    return content


def _sanitize_math_blocks(text: str) -> Tuple[str, int]:
    count = 0

    def clean_display(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "$$" + _clean_math_content(m.group(1)) + "$$"

    def clean_inline(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "$" + _clean_math_content(m.group(1)) + "$"

    text = _MATH_DISPLAY_RE.sub(clean_display, text)
    text = _MATH_INLINE_RE.sub(clean_inline, text)
    return text, count


def _sanitize_paren_bracket_math(text: str) -> Tuple[str, int]:
    count = 0

    def clean_bracket(m: re.Match) -> str:
        nonlocal count
        count += 1
        return r"\[" + _clean_math_content(m.group(1)) + r"\]"

    def clean_paren(m: re.Match) -> str:
        nonlocal count
        count += 1
        return r"\(" + _clean_math_content(m.group(1)) + r"\)"

    text = _BRACKET_DISPLAY_RE.sub(clean_bracket, text)
    text = _PAREN_INLINE_RE.sub(clean_paren, text)
    return text, count


def _sanitize_math_environments(text: str) -> Tuple[str, int]:
    def clean_env(m: re.Match) -> str:
        env_name = m.group(1)
        content = _clean_math_content(m.group(2))
        return f"\\begin{{{env_name}}}{content}\\end{{{env_name}}}"

    return _MATH_ENV_RE.subn(clean_env, text)


# ============================================================
# Text-mode escaping (prevents Missing $ inserted from _ ^)
# ============================================================

def _escape_caret_underscore_everywhere_outside_math(tex: str) -> tuple[str, int]:
    """
    Escape ^ and _ in text mode only (outside math).
    """
    if not tex:
        return tex, 0

    out: list[str] = []
    fixes = 0
    i = 0
    n = len(tex)

    state = "text"  # text|inline|display

    while i < n:
        # \(...\), \[...\]
        if tex.startswith(r"\(", i) and state == "text":
            state = "inline"; out.append(r"\("); i += 2; continue
        if tex.startswith(r"\)", i) and state == "inline":
            state = "text"; out.append(r"\)"); i += 2; continue
        if tex.startswith(r"\[", i) and state == "text":
            state = "display"; out.append(r"\["); i += 2; continue
        if tex.startswith(r"\]", i) and state == "display":
            state = "text"; out.append(r"\]"); i += 2; continue

        # leftover $ or $$ (should be rare after balancing)
        if tex[i] == "$":
            dbl = (i + 1 < n and tex[i + 1] == "$")
            if state == "text":
                state = "display" if dbl else "inline"
            else:
                if state == "inline" and not dbl: state = "text"
                elif state == "display" and dbl: state = "text"
            out.append("$$" if dbl else "$")
            i += 2 if dbl else 1
            continue

        ch = tex[i]
        if state == "text":
            if ch == "^":
                out.append(r"\textasciicircum{}"); fixes += 1; i += 1; continue
            if ch == "_":
                out.append(r"\_"); fixes += 1; i += 1; continue

        out.append(ch)
        i += 1

    return "".join(out), fixes


def _force_close_unclosed_bracket_math(tex: str) -> tuple[str, int]:
    opens = len(re.findall(r"\\\[", tex))
    closes = len(re.findall(r"\\\]", tex))
    if opens > closes:
        return tex + ("\n" + (r"\]" * (opens - closes)) + "\n"), (opens - closes)
    return tex, 0


def _handle_font_setup(tex: str, font_name: str) -> str:
    tex = tex.replace("\t", "  ")

    if re.search(r"\\setmainfont(?:\[[^\]]*\])?\{[^}]+\}", tex):
        tex = re.sub(
            r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
            rf"\1{font_name}\3",
            tex,
        )
    else:
        m = re.search(r"(\\usepackage(?:\[[^\]]*\])?\{fontspec\}\s*)", tex)
        if m:
            insert_at = m.end()
            tex = tex[:insert_at] + f"\\setmainfont{{{font_name}}}\n" + tex[insert_at:]

    return tex

def _ensure_hebrew_monospace(tex: str, mono_font: str = "Courier New") -> tuple[str, int]:
    """
    Polyglossia + Hebrew requires a monospace font that contains Hebrew
    if \\texttt or \\verb is used. Define \\hebrewfonttt.
    Returns (tex, fixes_count).
    """
    if not tex:
        return tex, 0

    # If already defined, do nothing
    if re.search(r"\\newfontfamily\\hebrewfonttt\{", tex):
        return tex, 0

    insert = f"\\newfontfamily\\hebrewfonttt{{{mono_font}}}\n"

    # Prefer inserting after polyglossia
    m = re.search(r"(\\usepackage(?:\[[^\]]*\])?\{polyglossia\}\s*)", tex)
    if m:
        at = m.end()
        return tex[:at] + insert + tex[at:], 1

    # Otherwise after fontspec
    m = re.search(r"(\\usepackage(?:\[[^\]]*\])?\{fontspec\}\s*)", tex)
    if m:
        at = m.end()
        return tex[:at] + insert + tex[at:], 1

    # Otherwise after \documentclass
    m = re.search(r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)", tex)
    if m:
        at = m.end()
        return tex[:at] + insert + tex[at:], 1

    # Fallback: prepend
    return insert + tex, 1



def _final_cleanup(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


# ============================================================
# Public API
# ============================================================

def clean_tex_robust(tex: str, font_name: str = "Arial") -> Tuple[str, str]:
    if not tex:
        return "", "No content to clean."

    original = tex
    fixes: Dict[str, int] = {}

    # Phase 1: normalize
    tex = _normalize_line_endings(tex)

    # Phase 2: Fix dangerous sequences
    tex, n = _fix_double_escaped_underscore(tex)
    if n: fixes["double_escaped_underscore_fixed"] = n

    tex, n = _fix_newline_underscore_in_text_mode(tex)
    if n: fixes["newline_underscore_textmode_fixed"] = n

    tex, n = _fix_textbackslash_sequences(tex)
    if n: fixes["textbackslash_fixed"] = n

    tex, n = _fix_escaped_dollars_in_commands(tex)
    if n: fixes["escaped_dollars_fixed"] = n

    tex, n = _fix_malformed_environments(tex)
    if n: fixes["malformed_environments_fixed"] = n

    tex, n = _fix_newline_escape_before_underscore(tex)
    if n: fixes["newline_underscore_fixed"] = n

    tex, n = _fix_newline_escape_before_caret(tex)
    if n: fixes["newline_caret_fixed"] = n

    tex, n = _fix_double_escaped_underscore(tex)
    if n: fixes["double_escaped_underscore_fixed"] = n

    # ✅ important: turn \\_ into \_ early
    tex, n = _fix_double_escaped_underscore(tex)
    if n: fixes["double_escaped_underscore_fixed"] = n

    # Phase 3: dollars -> normalized and converted to \( \) / \[ \]
    tex, n = _fix_multi_dollars(tex)
    if n: fixes["multi_dollars_fixed"] = n

    tex, n = _balance_math_delimiters(tex, prefer_paren_bracket=True)
    if n: fixes["math_delimiters_balanced"] = n

    tex, n = _close_unbalanced_inline_paren_math_global(tex)
    if n: fixes["inline_paren_math_closed_global"] = n

    # After your wrapping / balancing passes:
    tex, n = _close_unbalanced_inline_paren_math(tex)
    if n: fixes["global_inline_paren_math_closed"] = n

    # Phase 4: sanitize math in the form we now prefer: \( \) and \[ \]
    tex, n = _sanitize_paren_bracket_math(tex)
    if n: fixes["paren_bracket_math_sanitized"] = n

    tex, n = _fix_alignment_tokens_in_math(tex)
    if n: fixes["alignment_tokens_in_math_fixed"] = n

    # Also sanitize any leftover $ / $$ blocks (rare but possible)
    tex, n = _sanitize_math_blocks(tex)
    if n: fixes["math_blocks_sanitized"] = n

    tex, n = _sanitize_math_environments(tex)
    if n: fixes["math_environments_sanitized"] = n

    # Phase 5: general content repairs
    tex, n = _fix_undefined_commands(tex)
    if n: fixes["undefined_commands_fixed"] = n

    tex, n = _fix_invalid_delimiters(tex)
    if n: fixes["invalid_delimiters_fixed"] = n

    tex, n = _fix_itemize_blocks(tex)
    if n: fixes["itemize_blocks_fixed"] = n

    # Safer global balancing of \( \) and \[ \]
    tex, n = _balance_paren_bracket_math_globally(tex)
    if n: fixes["paren_bracket_math_globally_balanced"] = n

    # NEW: wrap mathy \item lines so _/^ don't break
    tex, n = _wrap_mathy_item_lines(tex)
    if n: fixes["mathy_item_lines_wrapped"] = n

    # Phase 6: fonts + text-mode escaping
    tex = _handle_font_setup(tex, font_name)

    tex, n = _fix_operator_set_braces_in_math(tex)
    if n: fixes["operator_set_braces_fixed"] = n

    tex, n = _escape_caret_underscore_everywhere_outside_math(tex)
    if n: fixes["text_caret_underscore_escaped"] = n

    tex, n = _ensure_hebrew_monospace(tex, mono_font=os.getenv("MATHGRADE_MONO_FONT", "Courier New"))
    if n: fixes["hebrew_monospace_defined"] = n

    tex, n = _wrap_item_math_lines(tex)
    if n: fixes["item_math_lines_wrapped"] = n

    tex, n = _force_close_unclosed_bracket_math(tex)
    if n: fixes["unclosed_bracket_math_closed"] = n

    # FINAL safety: guarantee no "\\_" survives in text mode
    tex, n = _final_fix_double_backslash_underscore_in_text(tex)
    if n: fixes["final_double_backslash_underscore_fixed"] = n

    tex, n = _wrap_operator_phrase_math_in_text(tex)
    if n: fixes["wrap_operator_phrase_math_in_text"] = n

    # Phase 7: final
    tex = _final_cleanup(tex)

    report_lines = ["LaTeX cleanup report:"]
    if not fixes:
        report_lines.append("  No issues found - document was already clean.")
    else:
        for k, v in fixes.items():
            report_lines.append(f"  {k}: {v}")

    report_lines.append(f"  Document changed: {tex != original}")
    return tex, "\n".join(report_lines)


# Backwards compatibility
# def clean_tex(tex: str) -> Tuple[str, Dict]:
#     cleaned, report = clean_tex_robust(tex)
#     report = {}
#
#     res = proofread_tex_with_ollama(tex=cleaned)
#     report["ollama_proofread"] = {
#         "used_ai": res.used_ai,
#         "ok": res.ok,
#         "reason": res.reason,
#         "meta": res.meta,
#     }
#     cleaned = res.tex  # only replaced if acceptable; otherwise original returned
#
#     return cleaned, {"report": report}

def clean_tex(tex: str, font_name: str = "Arial", *, ollama_proofread: bool = False, **kwargs):
    cleaned, report = clean_tex_robust(tex)

    if ollama_proofread:
        original = cleaned  # keep for "changed" flag

        res = proofread_tex_with_ollama(
            cleaned,
            model=kwargs.get("ollama_model"),
            base_url=kwargs.get("ollama_base_url"),
            timeout_s=kwargs.get("ollama_timeout_s", 300),
            retries=kwargs.get("ollama_retries", 1),
            num_predict=kwargs.get("ollama_num_predict", 1800),
            max_chars=kwargs.get("ollama_max_chars", 40_000),
            safe_mode=True,
        )

        cleaned = res.tex  # (module already returns original on failure/reject)

        # ✅ append to existing string report
        extra_lines = [
            "",
            "Ollama proofread:",
            f"  Used AI: {res.used_ai}",
            f"  OK: {res.ok}",
            f"  Reason: {res.reason}",
            f"  Document changed: {cleaned != original}",
        ]
        report = (report or "") + "\n".join(extra_lines)

    return cleaned, report