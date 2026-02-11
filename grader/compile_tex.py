# grader/compile_tex.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .tex_cleaner import clean_tex


@dataclass(frozen=True)
class CompileOutputs:
    compiled_tex: Path
    pdf: Path
    clean_report: Optional[Path] = None


def _copy_latex_logs(output_dir: Path, compiled_tex: Path) -> Path | None:
    """
    Copy the .log file into ./debug_logs so it's accessible even if output_dir is temp/AppData.
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
    Run subprocess command. On XeLaTeX failure, capture stdout and copy .log into ./debug_logs.
    Returns captured stdout.
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


def compile_tex_to_pdf(
    input_tex: Path,
    out_dir: Path,
    *,
    clean: bool = True,               # ✅ optional cleaning
    font_name: str = "Arial",
    passes: int = 2,
    xelatex: str = "xelatex",
    texinputs: Optional[list[Path]] = None,
) -> CompileOutputs:
    """
    Compile a .tex file to PDF using XeLaTeX.

    If clean=True:
      - Uses grader/tex_cleaner.py clean_tex_robust() (ALL cleaning is there)
      - Writes <stem>_clean.tex and <stem>_clean_report.txt into out_dir
      - Compiles the cleaned tex

    If clean=False:
      - Writes <stem>_raw.tex into out_dir (a copy of original)
      - Compiles the raw tex (no cleaning, no extra transforms)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _require_tool(xelatex)

    if not input_tex.exists():
        raise FileNotFoundError(f"Missing input_tex: {input_tex.resolve()}")

    src = input_tex.read_text(encoding="utf-8", errors="ignore")

    report_path: Optional[Path] = None
    if clean:

        cleaned, report = clean_tex(src, font_name=font_name, ollama_proofread=True)
        report_path = out_dir / f"{input_tex.stem}_clean_report.txt"
        report_path.write_text(report, encoding="utf-8")

        compiled_tex_path = out_dir / f"{input_tex.stem}_clean.tex"
        compiled_tex_path.write_text(cleaned, encoding="utf-8")
    else:
        # compile raw (no cleaning)
        compiled_tex_path = out_dir / f"{input_tex.stem}_raw.tex"
        compiled_tex_path.write_text(src, encoding="utf-8")

    # XeLaTeX output name follows compiled tex stem
    pdf_path = out_dir / f"{compiled_tex_path.stem}.pdf"

    # Ensure \input{} and other relative includes can be found.
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
                str(compiled_tex_path),
            ],
            env=env,
            output_dir=out_dir,
            compiled_tex=compiled_tex_path,
        )

    for _ in range(max(1, int(passes))):
        _xelatex_once()

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not created: {pdf_path.resolve()}")

    return CompileOutputs(compiled_tex=compiled_tex_path, pdf=pdf_path, clean_report=report_path)
