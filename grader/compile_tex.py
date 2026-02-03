# grader/compile_tex.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CompileOutputs:
    clean_tex: Path
    pdf: Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(cmd))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise


def _require_tool(exe: str) -> None:
    try:
        subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    except Exception:
        raise RuntimeError(
            f"Missing dependency: '{exe}' not found on PATH.\n"
            f"Install it and reopen your terminal/PyCharm so PATH updates.\n"
        )


def clean_tex_for_windows(tex: str, font_name: str = "Arial") -> str:
    """
    Minimal cleaning that helps XeLaTeX succeed on Windows and keeps bidi/RTL stable.
    - normalize tabs
    - force \setmainfont{<font_name>} if present
    - strip trailing whitespace
    """
    tex = tex.replace("\t", "  ")

    # Replace any \setmainfont[...]{} or \setmainfont{} with the requested font.
    tex = re.sub(
        r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
        rf"\1{font_name}\3",
        tex,
    )

    tex = "\n".join(line.rstrip() for line in tex.splitlines()) + "\n"
    return tex


def compile_tex_to_pdf(
    input_tex: Path,
    out_dir: Path,
    *,
    clean: bool = True,
    font_name: str = "Arial",
    passes: int = 2,
    xelatex: str = "xelatex",
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
    cleaned = clean_tex_for_windows(src, font_name=font_name) if clean else src

    clean_tex = out_dir / f"{input_tex.stem}_clean.tex"
    clean_tex.write_text(cleaned, encoding="utf-8")

    # XeLaTeX output name follows tex stem
    pdf_path = out_dir / f"{clean_tex.stem}.pdf"

    for _ in range(max(1, int(passes))):
        _run(
            [
                xelatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir}",
                str(clean_tex),
            ]
        )

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not created: {pdf_path.resolve()}")

    return CompileOutputs(clean_tex=clean_tex, pdf=pdf_path)
