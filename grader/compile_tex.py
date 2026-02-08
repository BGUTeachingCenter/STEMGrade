# grader/compile_tex.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .tex_cleaner import clean_tex


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
) -> None:
    """
    Run a subprocess command. On XeLaTeX failure, capture stdout and copy .log
    into ./debug_logs for easy access.
    """
    try:
        subprocess.run(
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
    Apply robust AI-oriented TeX cleaning (math delimiter normalization, text escaping, unicode)
    + Windows font/bidi stabilization.

    Returns:
        (cleaned_tex, human_readable_report)
    """
    cleaned, report = clean_tex(tex)  # robust cleaner from tex_cleaner.py
    cleaned = clean_tex_for_windows(cleaned, font_name=font_name)

    # Compact report for logs/debug (report is a dict)
    notes = report.get("notes") or []
    if not isinstance(notes, list):
        notes = [str(notes)]

    report_lines = [
        "TeX clean report:",
        f"  triple_dollars_fixed: {report.get('triple_dollars_fixed', 0)}",
        f"  bracket_math_converted: {report.get('bracket_math_converted', 0)}",
        f"  paren_math_converted: {report.get('paren_math_converted', 0)}",
        f"  env_math_sanitized: {report.get('env_math_sanitized', 0)}",
        f"  dollar_math_sanitized: {report.get('dollar_math_sanitized', 0)}",
        f"  set_notation_fixed: {report.get('set_notation_fixed', 0)}",
        f"  invalid_delimiters_fixed: {report.get('invalid_delimiters_fixed', 0)}",
    ]

    if notes:
        report_lines.append("  notes:")
        report_lines.extend([f"    - {str(n)}" for n in notes])

    return cleaned, "\n".join(report_lines) + "\n"



def compile_tex_to_pdf(
    input_tex: Path,
    out_dir: Path,
    *,
    clean: bool = True,
    font_name: str = "Arial",
    passes: int = 2,
    xelatex: str = "xelatex",
    texinputs: Optional[list[Path]] = None,
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
    if clean:
        cleaned, clean_report = clean_tex_robust(src, font_name=font_name)

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

    for _ in range(max(1, int(passes))):
        _run(
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

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not created: {pdf_path.resolve()}")

    return CompileOutputs(clean_tex=clean_tex_path, pdf=pdf_path)
