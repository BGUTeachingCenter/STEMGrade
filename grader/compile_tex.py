# compile_tex.py
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CompileResult:
    clean_tex: Path
    pdf: Optional[Path]
    docx: Optional[Path]


def require_tool(exe: str):
    """Raise a friendly error if exe isn't on PATH."""
    try:
        subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: '{exe}' not found on PATH.\n"
            f"Install it and reopen PyCharm/terminal so PATH updates.\n"
            f"Details: {e}\n"
        )


def run(cmd, cwd=None):
    """Run a command and print output on failure."""
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(map(str, cmd)))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise


def clean_tex_for_windows(tex: str, font_name: str = "Arial") -> str:
    """
    Minimal "polish" pass:
    - normalize tabs -> spaces
    - force a Windows-available font in \\setmainfont{...}
    - trim trailing whitespace
    """
    tex = tex.replace("\t", "  ")

    tex = re.sub(
        r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
        rf"\1{font_name}\3",
        tex,
    )

    tex = "\n".join(line.rstrip() for line in tex.splitlines()) + "\n"
    return tex


def compile_tex_file(
    input_tex: Path,
    out_dir: Path = Path("../build"),
    *,
    clean_suffix: str = "_clean",
    font_name: str = "Arial",
    xelatex: str = "xelatex",
    passes: int = 2,
    make_pdf: bool = True,
    make_docx: bool = True,
    pandoc: str = "pandoc",
    reference_docx: Optional[Path] = Path("reference.docx"),
) -> CompileResult:
    """
    Compile a .tex into:
      - cleaned .tex in out_dir
      - PDF (optional)
      - DOCX via pandoc (optional)

    Returns paths to outputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_tex.exists():
        raise FileNotFoundError(f"Can't find {input_tex.resolve()}")

    # Tool checks
    if make_pdf:
        require_tool(xelatex)
    if make_docx:
        require_tool(pandoc)

    stem = input_tex.stem
    clean_tex = out_dir / f"{stem}{clean_suffix}.tex"
    pdf_path = out_dir / f"{stem}{clean_suffix}.pdf" if make_pdf else None
    docx_path = out_dir / f"{stem}{clean_suffix}.docx" if make_docx else None

    # Read + clean
    src = input_tex.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_tex_for_windows(src, font_name=font_name)
    clean_tex.write_text(cleaned, encoding="utf-8")

    # 1) PDF compilation
    if make_pdf:
        print(f"Compiling PDF with XeLaTeX: {clean_tex.name}")
        for _ in range(max(1, passes)):
            run(
                [
                    xelatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={out_dir}",
                    str(clean_tex),
                ]
            )

        produced_pdf = out_dir / f"{clean_tex.stem}.pdf"
        if not produced_pdf.exists():
            raise RuntimeError("PDF not created. Check XeLaTeX logs in output dir.")

        # Normalize to our expected output name if different
        if produced_pdf != pdf_path:
            produced_pdf.replace(pdf_path)

    # 2) DOCX conversion
    if make_docx:
        print(f"Converting to DOCX with Pandoc: {clean_tex.name}")
        pandoc_cmd = [
            pandoc,
            str(clean_tex),
            "-o",
            str(docx_path),
            "--from=latex",
            "--to=docx",
            "--standalone",
            "--metadata",
            "lang=he",
        ]
        if reference_docx and reference_docx.exists():
            pandoc_cmd += ["--reference-doc", str(reference_docx)]
        run(pandoc_cmd)

        if not docx_path.exists():
            raise RuntimeError("DOCX not created. Check Pandoc output.")

    return CompileResult(clean_tex=clean_tex, pdf=pdf_path, docx=docx_path)
