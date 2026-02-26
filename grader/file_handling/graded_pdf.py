# grader/ai_grading/graded_pdf.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Tuple

from grader.file_handling.compile_tex import compile_tex_to_pdf


def _split_tex(tex_text: str) -> Tuple[str, str]:
    """
    Split a TeX document into (preamble_without_begin_document, body_without_end_document).
    If no \\begin{document} is found, treat entire file as body and preamble empty.
    """
    if not tex_text:
        return "", ""

    begin = tex_text.find(r"\begin{document}")
    end = tex_text.rfind(r"\end{document}")

    if begin == -1:
        # no document wrapper -> body only
        return "", tex_text.strip()

    preamble = tex_text[:begin].rstrip()

    body_start = begin + len(r"\begin{document}")
    if end == -1 or end < body_start:
        body = tex_text[body_start:].strip()
    else:
        body = tex_text[body_start:end].strip()

    return preamble, body


def _write_debug(out_dir: Path, name: str, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(text, encoding="utf-8", errors="replace")


def build_graded_pdf(
    *,
    qa_tex: Path,
    feedback_tex: Path,
    out_dir: Path,
    font_name: str = "Arial",
) -> Path:
    """
    New flow (TeX-first):
      1) Accept two TeX files (qa bundle TeX + feedback TeX)
      2) Unite them into out_dir/graded_union.tex
         - Reuse the QA preamble (macros/packages) to avoid missing commands
         - Append QA body, then \\clearpage, then feedback body
      3) Try compile_tex_to_pdf(union_tex)
      4) If compilation fails -> return union_tex path (fallback)

    Return:
      - PDF path on success
      - union .tex path on failure
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_text = qa_tex.read_text(encoding="utf-8", errors="replace")
    fb_text = feedback_tex.read_text(encoding="utf-8", errors="replace")

    qa_preamble, qa_body = _split_tex(qa_text)
    _fb_preamble, fb_body = _split_tex(fb_text)

    # Build union tex
    union_tex = out_dir / "graded_union.tex"

    # Ensure we have a document wrapper. Prefer QA preamble; if missing, create minimal.
    if qa_preamble.strip():
        preamble = qa_preamble
        # If QA preamble doesn't set font, we still try to set it safely (optional)
        # We'll only add setmainfont if fontspec is present AND setmainfont is not already used
        if "fontspec" in preamble and "setmainfont" not in preamble:
            preamble += f"\n\\setmainfont{{{font_name}}}\n"
    else:
        preamble = "\n".join([
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=2cm]{geometry}",
            r"\usepackage{fontspec}",
            rf"\setmainfont{{{font_name}}}",
            r"\usepackage{amsmath,amssymb,mathtools}",
            r"\usepackage{polyglossia}",
            r"\usepackage{bidi}",
            r"\setdefaultlanguage{hebrew}",
            r"\setotherlanguage{english}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.5em}",
        ])

    content = (
        preamble.strip()
        + "\n\n\\begin{document}\n\n"
        + qa_body.strip()
        + "\n\n\\clearpage\n\n"
        + fb_body.strip()
        + "\n\n\\end{document}\n"
    )

    union_tex.write_text(content, encoding="utf-8")

    # Try compile
    try:
        outputs = compile_tex_to_pdf(
            union_tex,
            out_dir,
            clean=False,
            font_name=font_name,
        )
        pdf_path = getattr(outputs, "pdf", None) or (out_dir / "graded_union.pdf")
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise RuntimeError("XeLaTeX finished but output PDF was not found.")

        return pdf_path

    except Exception as e:
        # Keep clear logs for debugging
        _write_debug(out_dir, "xelatex_union_error.txt", f"{e}\n\n{traceback.format_exc()}")
        _write_debug(out_dir, "graded_union.tex", content)  # ensures it exists even if partial write
        return union_tex