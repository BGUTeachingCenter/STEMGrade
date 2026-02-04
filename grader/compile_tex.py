# grader/compile_tex.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CompileOutputs:
    clean_tex: Path
    pdf: Path


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
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
    r"""Minimal cleaning that helps XeLaTeX succeed on Windows and keeps bidi/RTL stable.

    Notes:
      - normalize tabs
      - force \setmainfont{<font_name>} if present
      - strip trailing whitespace

    This docstring is a *raw* string to avoid Python "invalid escape sequence" warnings
    for LaTeX backslashes.
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
    cleaned = clean_tex_for_windows(src, font_name=font_name) if clean else src

    clean_tex = out_dir / f"{input_tex.stem}_clean.tex"
    clean_tex.write_text(cleaned, encoding="utf-8")

    # XeLaTeX output name follows tex stem
    pdf_path = out_dir / f"{clean_tex.stem}.pdf"

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
                f"-output-directory={out_dir}",
                str(clean_tex),
            ],
            env=env,
        )

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not created: {pdf_path.resolve()}")

    return CompileOutputs(clean_tex=clean_tex, pdf=pdf_path)
