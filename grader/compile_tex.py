# grader/compile_tex.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CompileOutputs:
    clean_tex: Path
    pdf: Path


def _copy_latex_logs(output_dir: Path, compiled_tex: Path) -> Path | None:
    """
    Copy the .log file into a local debug folder, so user can read it
    even if output_dir is inside AppData/Temp.

    NOTE: the .log file name follows the *compiled* tex stem.
    If we compile 'graded_feedback_clean.tex', the log will be
    'graded_feedback_clean.log'.
    """
    try:
        stem = compiled_tex.stem
        log_path = output_dir / f"{stem}.log"
        if not log_path.exists():
            return None

        debug_dir = Path.cwd() / "debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        copied = debug_dir / f"{stem}_{ts}.log"
        copied.write_text(
            log_path.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
        return copied
    except Exception:
        return None


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    output_dir: Path | None = None,
    compiled_tex: Path | None = None,
) -> str:
    """
    Run a subprocess command. On XeLaTeX failure, capture stdout and copy .log
    into ./debug_logs for easy access.

    Returns captured stdout text (useful for auto-retry heuristics).
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return p.stdout or ""
    except subprocess.CalledProcessError as e:
        copied = None
        if output_dir is not None and compiled_tex is not None:
            copied = _copy_latex_logs(output_dir, compiled_tex)

        extra = f"\n\n--- xelatex output (captured) ---\n{e.stdout}" if e.stdout else ""
        if copied:
            extra += f"\n\nSaved LaTeX log to: {copied}"

        raise RuntimeError(f"XeLaTeX failed: {e}{extra}") from e


def _require_tool(exe: str) -> None:
    try:
        subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: '{exe}' not found on PATH.\n"
            f"Install it and reopen your terminal/PyCharm so PATH updates.\n"
        ) from e


def clean_tex_for_windows(tex: str, font_name: str = "Arial") -> str:
    r"""Windows-focused normalization + font forcing (keeps bidi/RTL stable).

    Notes:
      - normalize tabs
      - force \setmainfont{<font_name>} if present (or insert if missing)
      - strip trailing whitespace
    """
    tex = tex.replace("\t", "  ")

    # Replace any \setmainfont[...]{} or \setmainfont{} with the requested font.
    if re.search(r"\\setmainfont(?:\[[^\]]*\])?\{[^}]+\}", tex):
        tex = re.sub(
            r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
            rf"\1{font_name}\3",
            tex,
        )
    else:
        # Insert \setmainfont{...} after \usepackage{fontspec} if possible.
        m = re.search(r"(\\usepackage(?:\[[^\]]*\])?\{fontspec\}\s*)", tex)
        if m:
            insert_at = m.end()
            tex = tex[:insert_at] + f"\\setmainfont{{{font_name}}}\n" + tex[insert_at:]

    tex = "\n".join(line.rstrip() for line in tex.splitlines()) + "\n"
    return tex


def clean_tex_robust(tex: str, font_name: str = "Arial") -> tuple[str, str]:
    """
    Use the robust tex_cleaner module for all cleaning.
    """
    from .tex_cleaner import clean_tex_robust as robust_clean

    return robust_clean(tex, font_name)


# =============================================================================
# NEW: robust $ / $$ delimiter balancer (state machine)
# =============================================================================

_BEGIN_ENV_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_END_ENV_RE = re.compile(r"\\end\{([a-zA-Z*]+)\}")


def _looks_like_display(math_text: str) -> bool:
    """Heuristic: decide whether math content is display-ish."""
    t = math_text.strip()
    if not t:
        return False
    if r"\\" in t or "&" in t:
        return True
    if any(cmd in t for cmd in (r"\sum", r"\int", r"\prod", r"\lim", r"\frac", r"\begin")):
        return True
    if len(t) > 45:
        return True
    return False


def _balance_math_delimiters(
    tex: str,
    *,
    prefer_paren_bracket: bool = True,  # $..$ -> \(...\), $$..$$ -> \[...\]
    force_inline_on_mismatch: bool = True,
) -> tuple[str, int]:
    """
    Walk document and normalize $ / $$ delimiters.
    Fixes $$...$ and $...$$ mismatches + unclosed math.
    Returns (new_tex, fixes_count).
    """
    if not tex:
        return tex, 0

    i, n = 0, len(tex)
    out: list[str] = []
    fixes = 0

    state = "text"  # "text" | "inline" | "display"
    math_buf: list[str] = []

    # Avoid touching dollars in verbatim-ish environments
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

        # Track verbatim-like environments (lightweight)
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

        # Escaped dollar
        if tex.startswith(r"\$", i):
            if state == "text":
                out.append(r"\$")
            else:
                math_buf.append(r"\$")
            i += 2
            continue

        # Dollar tokens
        if ch == "$":
            is_double = (i + 1 < n and tex[i + 1] == "$")
            token = "$$" if is_double else "$"

            if state == "text":
                state = "display" if is_double else "inline"
                if not prefer_paren_bracket:
                    out.append(token)
                i += 2 if is_double else 1
                continue

            # inline sees $$ -> likely $ ... $$
            if state == "inline" and token == "$$":
                flush_math("$")
                fixes += 1
                state = "text"
                i += 2
                continue

            # display sees $ -> likely $$ ... $
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

            # normal closing
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

            # fallback close
            close_as = "$" if state == "inline" else "$$"
            flush_math(close_as)
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

    # close if EOF inside math
    if state in ("inline", "display"):
        close_as = "$" if state == "inline" else "$$"
        flush_math(close_as)
        fixes += 1

    return "".join(out), fixes


# =============================================================================
# Main compile function
# =============================================================================


def compile_tex_to_pdf(
    input_tex: Path,
    out_dir: Path,
    *,
    clean: bool = True,
    font_name: str = "Arial",
    passes: int = 2,
    xelatex: str = "xelatex",
    texinputs: Optional[list[Path]] = None,
    balance_math_delims: bool = True,
    auto_retry_on_math_error: bool = True,
) -> CompileOutputs:
    """
    Compile a .tex file to PDF using XeLaTeX, writing outputs into out_dir.

    Returns:
      CompileOutputs(clean_tex=..., pdf=...)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _require_tool(xelatex)

    if not input_tex.exists():
        raise FileNotFoundError(f"Missing input_tex: {input_tex.resolve()}")

    src = input_tex.read_text(encoding="utf-8", errors="ignore")

    # ---- UPDATED: use robust cleaner ----
    clean_report = ""
    if clean:
        cleaned, clean_report = clean_tex_robust(src, font_name=font_name)

        # NEW: balance $ / $$ mismatches and convert to \(...\), \[...\]
        if balance_math_delims:
            cleaned2, nfix = _balance_math_delimiters(cleaned, prefer_paren_bracket=True)
            if nfix:
                cleaned = cleaned2
                clean_report += f"\nmath_delimiters_balanced: {nfix}\n"

        # Save a sidecar report to make debugging painless
        report_path = out_dir / f"{input_tex.stem}_clean_report.txt"
        report_path.write_text(clean_report, encoding="utf-8")
    else:
        cleaned = src

    clean_tex_path = out_dir / f"{input_tex.stem}_clean.tex"
    clean_tex_path.write_text(cleaned, encoding="utf-8")

    # XeLaTeX output name follows compiled tex stem
    pdf_path = out_dir / f"{clean_tex_path.stem}.pdf"

    # Ensure \input{} and other relative includes can be found even if we compile from out_dir.
    # Add the original TeX folder (and any extra texinputs) to TEXINPUTS.
    env = os.environ.copy()
    tex_paths = [input_tex.parent]
    if texinputs:
        tex_paths.extend([p for p in texinputs if p])

    prepend = os.pathsep.join(str(p.resolve()) for p in dict.fromkeys(tex_paths))
    existing = env.get("TEXINPUTS", "")
    env["TEXINPUTS"] = prepend + (os.pathsep + existing if existing else "") + os.pathsep

    def _xelatex_once() -> str:
        return _run(
            [
                xelatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={str(out_dir)}",
                str(clean_tex_path),
            ],
            env=env,
            output_dir=out_dir,
            compiled_tex=clean_tex_path,
        )

    # First attempt (normal passes)
    try:
        for _ in range(max(1, int(passes))):
            _xelatex_once()
    except RuntimeError as e:
        msg = str(e)

        # Auto-retry for math-delimiter failure class
        if auto_retry_on_math_error and (
            "Display math should end with $$" in msg
            or "Missing $ inserted" in msg
            or "Extra }, or forgotten $" in msg
        ):
            rebalanced, nfix2 = _balance_math_delimiters(
                cleaned,
                prefer_paren_bracket=True,
                force_inline_on_mismatch=False,  # keep display when unsure
            )
            if nfix2:
                cleaned = rebalanced
                clean_tex_path.write_text(cleaned, encoding="utf-8")

                # try again (same number of passes)
                for _ in range(max(1, int(passes))):
                    _xelatex_once()
            else:
                raise
        else:
            raise

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not created: {pdf_path.resolve()}")

    return CompileOutputs(clean_tex=clean_tex_path, pdf=pdf_path)
