import re
import subprocess
from pathlib import Path


# ---------- CONFIG ----------
INPUT_TEX = Path("012.tex")          # your source
OUT_DIR = Path("build")              # outputs go here
CLEAN_TEX = OUT_DIR / "012_clean.tex"
OUT_PDF = OUT_DIR / "012_clean.pdf"
OUT_DOCX = OUT_DIR / "012_clean.docx"
REFERENCE_DOCX = Path("reference.docx")  # optional: RTL Word template


# ---------- HELPERS ----------
def require_tool(exe: str):
    """Raise a friendly error if exe isn't on PATH."""
    try:
        subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    except Exception:
        raise RuntimeError(
            f"Missing dependency: '{exe}' not found on PATH.\n"
            f"Install it and reopen PyCharm/terminal so PATH updates.\n"
        )


def run(cmd, cwd=None):
    """Run a command and print output on failure."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(cmd))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise


def clean_tex_for_windows(tex: str) -> str:
    """
    Minimal "polish" pass:
    - ensure a Windows-available Hebrew font (Arial by default)
    - normalize tabs -> spaces
    - trim trailing whitespace
    - keep your RTL + bidi setup intact
    """
    # Normalize tabs
    tex = tex.replace("\t", "  ")

    # Replace \setmainfont[Script=Hebrew]{...} with Arial (safer on Windows)
    tex = re.sub(
        r"(\\setmainfont(?:\[[^\]]*\])?\{)([^}]+)(\})",
        r"\1Arial\3",
        tex
    )

    # Remove trailing spaces
    tex = "\n".join(line.rstrip() for line in tex.splitlines()) + "\n"

    return tex


# ---------- PIPELINE ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check required tools
    require_tool("xelatex")
    require_tool("pandoc")

    if not INPUT_TEX.exists():
        raise FileNotFoundError(f"Can't find {INPUT_TEX.resolve()}")

    # Read and clean
    src = INPUT_TEX.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_tex_for_windows(src)
    CLEAN_TEX.write_text(cleaned, encoding="utf-8")

    # 1) Compile PDF (2 passes helps references/layout)
    # XeLaTeX writes outputs into OUT_DIR by using -output-directory
    print("Compiling PDF with XeLaTeX...")
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={OUT_DIR}", str(CLEAN_TEX)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={OUT_DIR}", str(CLEAN_TEX)])

    if not OUT_PDF.exists():
        # XeLaTeX output name matches .tex stem
        # e.g., 012_clean.tex -> 012_clean.pdf
        raise RuntimeError("PDF not created. Check XeLaTeX logs in build/.")

    print(f"PDF created: {OUT_PDF.resolve()}")

    # 2) Convert to DOCX via Pandoc
    print("Converting to DOCX with Pandoc...")

    pandoc_cmd = [
        "pandoc",
        str(CLEAN_TEX),
        "-o", str(OUT_DOCX),
        "--from=latex",
        "--to=docx",
        "--standalone",
        "--metadata", "lang=he",
    ]

    # Use reference.docx if you created one (strongly recommended for RTL)
    if REFERENCE_DOCX.exists():
        pandoc_cmd += ["--reference-doc", str(REFERENCE_DOCX)]

    run(pandoc_cmd)

    if not OUT_DOCX.exists():
        raise RuntimeError("DOCX not created. Check Pandoc output.")

    print(f"DOCX created: {OUT_DOCX.resolve()}")
    print("\nDone.\n"
          f"Open PDF : {OUT_PDF}\n"
          f"Open DOCX: {OUT_DOCX}\n")


if __name__ == "__main__":
    main()
