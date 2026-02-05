# grader/bundle_writer.py
"""Generate the LaTeX source that creates the final Q/A bundle PDF.

The bundle is a single PDF that contains, per question/part:
  1) The relevant page range from the reference (question + official solution)
  2) A PDF rendering of the student's answer snippet

The output of this module is a `.tex` file. Compiling it (XeLaTeX) produces the
final bundle PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .reference_ranges import Key


def write_bundle_tex_from_reference_snippets(
    out_tex: Path,
    *,
    reference_snippets: Dict[Key, str],
    answer_pdfs: Dict[Key, Optional[Path]],
    font_name: str,
    title: str,
) -> None:
    """Write a LaTeX document that embeds reference *LaTeX snippets* + student answer PDFs.

    This is the PDF-bundle path when the reference is a .tex file (not a PDF).
    We directly typeset the reference snippet inside the bundle document and
    then include the student's rendered answer PDF for that part.
    """

    keys = sorted(reference_snippets.keys(), key=lambda k: (k[0], k[1]))

    tex: List[str] = []
    tex.append("\\documentclass[12pt]{article}\n")
    tex.append("\\usepackage[a4paper,margin=1.7cm]{geometry}\n")
    tex.append("\\usepackage{hyperref}\n")
    tex.append("\\usepackage{bookmark}\n")
    tex.append("\\usepackage{pdfpages}\n")
    tex.append("\\usepackage{xcolor}\n")
    tex.append("\\usepackage{amsmath,amssymb,mathtools}\n")
    tex.append("\\usepackage{fontspec}\n")
    tex.append("\\usepackage{bidi}\n")
    tex.append(f"\\setmainfont[Script=Hebrew]{{{font_name}}}\n")
    tex.append(f"\\setmonofont{{{font_name}}}\n")
    tex.append("\\setRTL\n")
    tex.append("\\setlength{\\parskip}{0.6em}\n")
    tex.append("\\setlength{\\parindent}{0pt}\n")
    tex.append("\\begin{document}\n")
    tex.append(f"\\section*{{{title}}}\n")
    tex.append("\\tableofcontents\n\\newpage\n")

    for (qnum, part) in keys:
        section_title = f"שאלה {qnum}" + (f"({part})" if part else "")
        tex.append(f"\\section{{{section_title}}}\n")

        tex.append("\\subsection*{Reference (question + official solution)}\n")
        ref_block = (reference_snippets.get((qnum, part)) or "").strip()
        if not ref_block:
            tex.append("\\textcolor{red}{Missing reference snippet for this part.}\\par\n")
        else:
            # Keep as raw LaTeX. Ensure separation from surrounding sectioning.
            tex.append("\\begingroup\n")
            tex.append(ref_block)
            tex.append("\n\\endgroup\n")

        tex.append("\\subsection*{Student answer (rendered)}\n")
        ap = answer_pdfs.get((qnum, part)) or answer_pdfs.get((qnum, ""))
        if ap is None:
            tex.append("\\textcolor{red}{Could not compile student answer for this part.}\\par\n")
        else:
            ap_tex = str(ap).replace("\\", "/")
            tex.append(f"\\includepdf[pages=-,pagecommand={{}}]{{{ap_tex}}}\n")

        tex.append("\\newpage\n")

    tex.append("\\end{document}\n")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("".join(tex), encoding="utf-8")


def write_bundle_tex(
    out_tex: Path,
    *,
    reference_pdf: Path,
    ref_ranges: Dict[Key, Tuple[int, int]],
    answer_pdfs: Dict[Key, Optional[Path]],
    font_name: str,
    title: str,
) -> None:
    """Write a LaTeX document that embeds the reference + answer PDFs.

    Args:
        out_tex: Path to write the .tex file.
        reference_pdf: The cleaned reference PDF.
        ref_ranges: (qnum, part) -> (start_page, end_page) ranges into reference_pdf.
        answer_pdfs: (qnum, part) -> path to per-snippet compiled answer PDFs.
        font_name: Font used for Hebrew RTL typesetting.
        title: Top-level title for the bundle document.
    """

    ref_pdf_tex = str(reference_pdf).replace("\\", "/")
    keys = sorted(ref_ranges.keys(), key=lambda k: (k[0], k[1]))

    tex: List[str] = []
    # IMPORTANT: use real newlines ("\n"), not the literal two characters "\\n".
    tex.append("\\documentclass[12pt]{article}\n")
    tex.append("\\usepackage[a4paper,margin=1.7cm]{geometry}\n")
    tex.append("\\usepackage{hyperref}\n")
    tex.append("\\usepackage{bookmark}\n")
    tex.append("\\usepackage{pdfpages}\n")
    tex.append("\\usepackage{xcolor}\n")
    tex.append("\\usepackage{fontspec}\n")
    tex.append("\\usepackage{bidi}\n")
    tex.append(f"\\setmainfont[Script=Hebrew]{{{font_name}}}\n")
    tex.append(f"\\setmonofont{{{font_name}}}\n")
    tex.append("\\setRTL\n")
    tex.append("\\setlength{\\parskip}{0.6em}\n")
    tex.append("\\setlength{\\parindent}{0pt}\n")
    tex.append("\\begin{document}\n")
    tex.append(f"\\section*{{{title}}}\n")
    tex.append("\\tableofcontents\n\\newpage\n")

    for (qnum, part) in keys:
        start, end = ref_ranges[(qnum, part)]
        section_title = f"שאלה {qnum}" + (f"({part})" if part else "")
        tex.append(f"\\section{{{section_title}}}\n")

        tex.append("\\subsection*{Reference (question + official solution)}\n")
        tex.append(f"\\includepdf[pages={{{start}-{end}}},pagecommand={{}}]{{{ref_pdf_tex}}}\n")

        tex.append("\\subsection*{Student answer (rendered)}\n")
        ap = answer_pdfs.get((qnum, part))
        if ap is None:
            # Fallback: if the student's file didn't split parts, try whole-question snippet.
            ap = answer_pdfs.get((qnum, ""))

        if ap is None:
            tex.append("\\textcolor{red}{Could not compile student answer for this part.}\n")
        else:
            ap_tex = str(ap).replace("\\", "/")
            tex.append(f"\\includepdf[pages=-,pagecommand={{}}]{{{ap_tex}}}\n")

        tex.append("\\newpage\n")

    tex.append("\\end{document}\n")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("".join(tex), encoding="utf-8")
