# qa_bundle.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

from .compile_tex import compile_tex_file


Key = Tuple[int, str]  # (qnum, part-letter)


@dataclass
class BundleOutputs:
    bundle_tex: Path
    bundle_pdf: Path
    student_clean_pdf: Path
    ref_ranges: Dict[Key, Tuple[int, int]]
    student_ranges: Dict[Key, Tuple[int, int]]


# --- Detection helpers --------------------------------------------------------

_HEB_LETTER_RE = r"[א-ת]"
_MARK_RE = re.compile(rf"שאלה\s+(\d+)\(({_HEB_LETTER_RE})\)")

def _page_text(doc: fitz.Document, i: int) -> str:
    # "text" is usually best for searching markers
    return doc[i].get_text("text") or ""


def find_question_markers(pdf_path: Path) -> List[Tuple[int, str, int]]:
    """
    Returns a list of (qnum, part, page_index_0based) sorted by page.
    Looks for markers like: 'שאלה 3(ב)'
    """
    doc = fitz.open(str(pdf_path))
    markers: Dict[Key, int] = {}

    for i in range(doc.page_count):
        t = _page_text(doc, i)
        for m in _MARK_RE.finditer(t):
            qnum = int(m.group(1))
            part = m.group(2)
            key = (qnum, part)
            # keep earliest page where marker appears
            markers.setdefault(key, i)

    # sort by page
    out = [(k[0], k[1], p) for k, p in markers.items()]
    out.sort(key=lambda x: x[2])
    return out


def build_page_ranges(
    markers: List[Tuple[int, str, int]],
    total_pages: int,
) -> Dict[Key, Tuple[int, int]]:
    """
    Converts marker pages into 1-based inclusive page ranges for pdfpages.
    Range is from marker page to the page before next marker.
    """
    ranges: Dict[Key, Tuple[int, int]] = {}
    for idx, (qnum, part, p0) in enumerate(markers):
        start0 = p0
        end0 = (markers[idx + 1][2] - 1) if idx + 1 < len(markers) else (total_pages - 1)
        # convert to 1-based
        ranges[(qnum, part)] = (start0 + 1, end0 + 1)
    return ranges


# --- LaTeX bundle builder -----------------------------------------------------

def _tex_path(p: Path) -> str:
    # LaTeX likes forward slashes even on Windows
    return str(p).replace("\\", "/")


def write_bundle_tex(
    out_tex: Path,
    reference_pdf: Path,
    student_pdf: Path,
    ref_ranges: Dict[Key, Tuple[int, int]],
    student_ranges: Dict[Key, Tuple[int, int]],
    *,
    title: str = "חוברת התאמה: שאלות + פתרונות + תשובות סטודנט/ית",
) -> None:
    """
    Writes a clean bundle TeX that includes PDF pages directly via pdfpages.
    """
    # Build a deterministic order: sort by qnum then part letter
    keys = sorted(ref_ranges.keys(), key=lambda k: (k[0], k[1]))

    ref_pdf_tex = _tex_path(reference_pdf)
    stu_pdf_tex = _tex_path(student_pdf)

    lines: List[str] = []
    lines.append(r"\documentclass[11pt]{article}" "\n")
    lines.append(r"\usepackage[a4paper,margin=1.5cm]{geometry}" "\n")
    lines.append(r"\usepackage{hyperref}" "\n")
    lines.append(r"\usepackage{bookmark}" "\n")
    lines.append(r"\usepackage{pdfpages}" "\n")
    lines.append(r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}" "\n")
    lines.append(r"\begin{document}" "\n\n")
    lines.append(rf"\section*{{{title}}}" "\n")
    lines.append(r"\tableofcontents" "\n")
    lines.append(r"\newpage" "\n\n")

    for (qnum, part) in keys:
        r_start, r_end = ref_ranges[(qnum, part)]
        s_range = student_ranges.get((qnum, part))
        # Section title in Hebrew marker format
        sec_title = f"שאלה {qnum}({part})"

        lines.append(rf"\section{{{sec_title}}}" "\n")

        lines.append(r"\subsection*{מבחן (נוסח שאלה + פתרון רשמי)}" "\n")
        lines.append(
            rf"\includepdf[pages={r_start}-{r_end},pagecommand={{}}]{{{ref_pdf_tex}}}" "\n"
        )

        lines.append(r"\subsection*{הגשת הסטודנט/ית (כפי שהודפסה)}" "\n")
        if s_range:
            s_start, s_end = s_range
            lines.append(
                rf"\includepdf[pages={s_start}-{s_end},pagecommand={{}}]{{{stu_pdf_tex}}}" "\n"
            )
        else:
            lines.append(
                r"\noindent\textbf{לא נמצאו סימוני 'שאלה X(א/ב)' בקובץ הסטודנט/ית, לכן לא ניתן לחלץ טווח עמודים אוטומטית.}\par"
                "\n"
                r"\noindent אפשר לפתור זאת ע״י הוספת כותרות 'שאלה 1(א)' וכו׳ במסמך ה-LaTeX או ע״י מיפוי ידני.\par"
                "\n\n"
            )

        lines.append(r"\newpage" "\n\n")

    lines.append(r"\end{document}" "\n")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("".join(lines), encoding="utf-8")


# --- Public API ---------------------------------------------------------------

def generate_qa_bundle_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path = Path("build"),
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> BundleOutputs:
    """
    1) Compiles student_tex -> out_dir/<student_stem>_clean.pdf using compile_tex_file()
    2) Detects question markers and builds page ranges for both:
        - reference_pdf
        - student_clean_pdf
    3) Writes out_dir/<bundle_stem>.tex that includes the pages
    4) Compiles it -> out_dir/<bundle_stem>.pdf
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reference_pdf.exists():
        raise FileNotFoundError(f"Missing reference PDF: {reference_pdf.resolve()}")
    if not student_tex.exists():
        raise FileNotFoundError(f"Missing student TeX: {student_tex.resolve()}")

    # Step 1: compile student
    student_result = compile_tex_file(
        input_tex=student_tex,
        out_dir=out_dir,
        make_pdf=True,
        make_docx=False,
        font_name=font_name,
        passes=2,
    )
    if not student_result.pdf or not student_result.pdf.exists():
        raise RuntimeError("Failed to compile student PDF. Check build logs.")
    student_pdf = student_result.pdf

    # Step 2: find markers & ranges
    ref_doc = fitz.open(str(reference_pdf))
    stu_doc = fitz.open(str(student_pdf))

    ref_markers = find_question_markers(reference_pdf)
    stu_markers = find_question_markers(student_pdf)

    ref_ranges = build_page_ranges(ref_markers, ref_doc.page_count)
    stu_ranges = build_page_ranges(stu_markers, stu_doc.page_count)

    # Step 3: write bundle TeX
    bundle_tex = out_dir / f"{bundle_stem}.tex"
    write_bundle_tex(
        out_tex=bundle_tex,
        reference_pdf=reference_pdf,
        student_pdf=student_pdf,
        ref_ranges=ref_ranges,
        student_ranges=stu_ranges,
    )

    # Step 4: compile bundle
    bundle_result = compile_tex_file(
        input_tex=bundle_tex,
        out_dir=out_dir,
        make_pdf=True,
        make_docx=False,
        font_name=font_name,
        passes=2,
    )

    if not bundle_result.pdf or not bundle_result.pdf.exists():
        raise RuntimeError("Failed to compile bundle PDF. Check build logs.")

    # normalize output name: compile_tex_file makes <stem>_clean.pdf
    produced_pdf = bundle_result.pdf
    final_pdf = out_dir / f"{bundle_stem}.pdf"
    if produced_pdf != final_pdf:
        if final_pdf.exists():
            final_pdf.unlink()
        produced_pdf.replace(final_pdf)

    return BundleOutputs(
        bundle_tex=bundle_tex,
        bundle_pdf=final_pdf,
        student_clean_pdf=student_pdf,
        ref_ranges=ref_ranges,
        student_ranges=stu_ranges,
    )
