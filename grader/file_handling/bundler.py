from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from grader.ai_grading.payloads import build_payloads
from grader.file_handling.answer_render import compile_student_answer_pdfs
from grader.file_handling.compile_tex_to_pdf import compile_tex_to_pdf
from grader.file_handling.part_normalize import normalize_part
from grader.file_handling.pdf_cleanse import CleanseReport, cleanse_test_pdf
from grader.file_handling.reference_ranges import Key, find_reference_ranges
from grader.file_handling.student_tex import parse_student_tex_answers


@dataclass(frozen=True)
class QABundleOutputs:
    bundle_tex: Path
    bundle_pdf: Optional[Path]
    student_clean_pdf: Path
    reference_clean_pdf: Path
    cleanse_report: CleanseReport
    ref_ranges: Dict[Key, Tuple[int, int]]
    student_ranges: Dict[Key, Tuple[int, int]]
    student_answer_pdfs: Dict[Key, Optional[Path]]


def generate_bundle(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
    compile_pdf: bool = False,
) -> QABundleOutputs:
    """
    Public entry point.

    Supported reference inputs:
      - .pdf
      - .tex / .txt

    By default, generates only the bundle .tex.
    Set compile_pdf=True to also compile the bundle PDF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = reference_tex.suffix.lower()
    if suffix == ".pdf":
        return _generate_from_reference_pdf(
            reference_pdf=reference_tex,
            student_tex=student_tex,
            out_dir=out_dir,
            bundle_stem=bundle_stem,
            font_name=font_name,
            compile_pdf=compile_pdf,
        )

    if suffix in {".tex", ".txt"}:
        return _generate_from_reference_tex(
            reference_tex=reference_tex,
            student_tex=student_tex,
            out_dir=out_dir,
            bundle_stem=bundle_stem,
            font_name=font_name,
            compile_pdf=compile_pdf,
        )

    raise ValueError(
        f"Unsupported reference type: {reference_tex.name}. Expected .pdf or .tex/.txt"
    )


def _norm_key(k: Key) -> Key:
    q, p = k
    return int(q), normalize_part(p)


def _union_keys(*dicts: Dict[Key, object]) -> list[Key]:
    keys: set[Key] = set()
    for d in dicts:
        for k in d.keys():
            keys.add(_norm_key(k))
    return sorted(keys, key=lambda x: (x[0], x[1]))


def _tex_path(path: Path) -> str:
    return path.as_posix()


def _tex_escape_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _sort_key(k: Key):
    qnum, part = k
    try:
        qnum_sort = int(qnum)
    except Exception:
        qnum_sort = str(qnum)
    part_sort = "" if part in (None, "") else str(part)
    return qnum_sort, part_sort


def _compile_bundle_pdf(
    *,
    bundle_tex: Path,
    out_dir: Path,
    font_name: str,
    texinputs: list[Path],
    bundle_stem: str,
) -> Path:
    bundle_pdf = compile_tex_to_pdf(
        bundle_tex,
        out_dir,
        clean=False,
        font_name=font_name,
        passes=2,
        texinputs=texinputs,
    ).pdf

    normalized = out_dir / f"{bundle_stem}.pdf"
    if bundle_pdf.exists() and bundle_pdf != normalized:
        try:
            bundle_pdf.replace(normalized)
            bundle_pdf = normalized
        except Exception:
            pass

    if not bundle_pdf.exists():
        raise RuntimeError("Bundle PDF was not created. Check LaTeX logs in build/.")

    return bundle_pdf


def _write_bundle_tex_from_pdf_reference(
    out_tex: Path,
    *,
    title: str,
    font_name: str,
    reference_pdf: Path,
    ref_ranges: Dict[Key, Tuple[int, int]],
    answer_pdfs: Dict[Key, Optional[Path]],
) -> None:
    """
    PDF-reference mode:
    include reference PDF page ranges + rendered student answer PDFs.
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    keys = _union_keys(ref_ranges, answer_pdfs)
    ref_pdf_tex = _tex_path(reference_pdf)

    tex: list[str] = []
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

    for qnum, part in keys:
        section_title = f"שאלה {qnum}" + (f"({part})" if part else "")
        tex.append(f"\\section{{{section_title}}}\n")

        tex.append("\\subsection*{Reference (question + official solution)}\n")
        rng = ref_ranges.get((qnum, part)) or ref_ranges.get((qnum, ""))
        if rng:
            start, end = rng
            tex.append(
                f"\\includepdf[pages={{{start}-{end}}},pagecommand={{}}]{{{ref_pdf_tex}}}\n"
            )
        else:
            tex.append("\\textcolor{red}{Missing reference page-range for this part.}\\par\n")

        tex.append("\\subsection*{Student answer (rendered)}\n")
        answer_pdf = answer_pdfs.get((qnum, part)) or answer_pdfs.get((qnum, ""))
        if answer_pdf is None:
            tex.append("\\textcolor{red}{Could not compile student answer for this part.}\\par\n")
        else:
            tex.append(
                f"\\includepdf[pages=-,pagecommand={{}}]{{{_tex_path(answer_pdf)}}}\n"
            )

        tex.append("\\newpage\n")

    tex.append("\\end{document}\n")
    out_tex.write_text("".join(tex), encoding="utf-8")


def _write_bundle_tex_inline_answers(
    out_tex: Path,
    *,
    title: str,
    font_name: str,
    reference_snippets: Dict[Key, str],
    student_answers: Dict[Key, str],
) -> None:
    """
    Fast TeX-first mode:
    include reference snippets + raw student LaTeX inline.
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    ordered_keys = sorted(
        set(reference_snippets.keys()) | set(student_answers.keys()),
        key=_sort_key,
    )

    parts: list[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{iftex}",
        r"\usepackage{fontspec}",
        rf"\setmainfont{{{font_name}}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\begin{document}",
        rf"{{\LARGE \textbf{{{_tex_escape_text(title)}}}}}\par",
        r"\vspace{1em}",
    ]

    for qnum, part in ordered_keys:
        label = f"Q{qnum}" + (f"({part})" if part else "")
        ref_block = (reference_snippets.get((qnum, part)) or "").strip()
        student_block = (student_answers.get((qnum, part)) or "").strip()

        parts.extend([
            r"\hrule",
            r"\vspace{0.8em}",
            rf"{{\large \textbf{{{_tex_escape_text(label)}}}}}\par",
            r"\vspace{0.5em}",
            r"{\bfseries Reference}\par",
            ref_block if ref_block else r"{\color{red}No reference content found.}",
            r"\vspace{0.8em}",
            r"{\bfseries Student answer}\par",
            student_block if student_block else r"{\color{red}No student answer found for this part.}",
            r"\vspace{1.2em}",
        ])

    parts.append(r"\end{document}")
    out_tex.write_text("\n".join(parts), encoding="utf-8")


def _generate_from_reference_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str,
    font_name: str,
    compile_pdf: bool,
) -> QABundleOutputs:
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    ref_ranges = {
        _norm_key(k): v
        for k, v in find_reference_ranges(reference_clean_pdf, out_dir).items()
    }
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed.\n"
            "Check build/debug_reference_pages.txt for extracted text.\n"
            "If it is empty, your PDF may be scanned (no selectable text)."
        )

    student_answers, student_ranges = parse_student_tex_answers(student_tex, out_dir)
    student_answers = {_norm_key(k): v for k, v in student_answers.items()}
    student_ranges = {_norm_key(k): v for k, v in student_ranges.items()}

    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX.\n"
            "Check build/debug_student_tex_parts.txt."
        )

    student_clean_pdf = compile_tex_to_pdf(
        student_tex,
        out_dir,
        clean=True,
        font_name=font_name,
        passes=2,
        texinputs=[student_tex.parent],
    ).pdf

    answer_pdfs = compile_student_answer_pdfs(
        student_answers,
        out_dir,
        font_name,
        clean=True,
    )
    answer_pdfs = {_norm_key(k): v for k, v in answer_pdfs.items()}

    bundle_tex = out_dir / f"{bundle_stem}.tex"
    _write_bundle_tex_from_pdf_reference(
        bundle_tex,
        title="Q/A Bundle (Reference pages + Rendered student answers)",
        font_name=font_name,
        reference_pdf=reference_clean_pdf,
        ref_ranges=ref_ranges,
        answer_pdfs=answer_pdfs,
    )

    bundle_pdf: Optional[Path] = None
    if compile_pdf:
        bundle_pdf = _compile_bundle_pdf(
            bundle_tex=bundle_tex,
            out_dir=out_dir,
            font_name=font_name,
            texinputs=[bundle_tex.parent],
            bundle_stem=bundle_stem,
        )

    return QABundleOutputs(
        bundle_tex=bundle_tex,
        bundle_pdf=bundle_pdf,
        student_clean_pdf=student_clean_pdf,
        reference_clean_pdf=reference_clean_pdf,
        cleanse_report=cleanse_report,
        ref_ranges=ref_ranges,
        student_ranges=student_ranges,
        student_answer_pdfs=answer_pdfs,
    )


def _generate_from_reference_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str,
    font_name: str,
    compile_pdf: bool,
) -> QABundleOutputs:
    """
    Fast TeX-first path:
    - uses build_payloads_from_reference_tex(...) as the source of truth
    - extracts reference snippets + raw student LaTeX from payloads
    - writes one bundle .tex with inline student answers
    - compiles only if compile_pdf=True
    """
    payload_out = out_dir / "bundle_payloads"
    payload_out.mkdir(parents=True, exist_ok=True)

    _, items = build_payloads(
        reference_tex=reference_tex,
        student_tex=student_tex,
        out_dir=payload_out,
        default_max_points=0.0,
    )

    reference_snippets: Dict[Key, str] = {}
    student_answers: Dict[Key, str] = {}
    student_ranges: Dict[Key, Tuple[int, int]] = {}

    for item in items:
        data = json.loads(item.payload_path.read_text(encoding="utf-8"))
        key = _norm_key((data["key"]["qnum"], data["key"]["part"]))

        question_text = (data.get("reference", {}).get("question_text") or "").strip()
        solution_text = (data.get("reference", {}).get("solution_text") or "").strip()
        reference_snippets[key] = "\n\n".join(
            part for part in [question_text, solution_text] if part
        ).strip()

        student_answers[key] = (data.get("student", {}).get("latex_raw") or "").strip()

    if not student_answers:
        raise RuntimeError("No student answers found in generated payloads (reference_tex path).")

    bundle_tex = out_dir / f"{bundle_stem}.tex"
    _write_bundle_tex_inline_answers(
        bundle_tex,
        title="Q/A Bundle (Reference TeX + Inline student answers)",
        font_name=font_name,
        reference_snippets=reference_snippets,
        student_answers=student_answers,
    )

    bundle_pdf: Optional[Path] = None
    if compile_pdf:
        bundle_pdf = _compile_bundle_pdf(
            bundle_tex=bundle_tex,
            out_dir=out_dir,
            font_name=font_name,
            texinputs=[bundle_tex.parent, reference_tex.parent, student_tex.parent],
            bundle_stem=bundle_stem,
        )

    placeholder_pdf = out_dir / "reference_tex_placeholder.pdf"
    student_placeholder_pdf = out_dir / "student_tex_placeholder.pdf"

    cleanse_report = CleanseReport(
        input_pdf=placeholder_pdf,
        output_pdf=placeholder_pdf,
        removed_pages_1based=(),
        kept_pages_1based=(),
        reason_lines=("Reference provided as .tex; PDF cleansing step skipped.",),
        debug_report_path=None,
    )

    return QABundleOutputs(
        bundle_tex=bundle_tex,
        bundle_pdf=bundle_pdf,
        student_clean_pdf=student_placeholder_pdf,
        reference_clean_pdf=placeholder_pdf,
        cleanse_report=cleanse_report,
        ref_ranges={},
        student_ranges=student_ranges,
        student_answer_pdfs={},
    )