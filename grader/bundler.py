from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

from .ai_grading.payloads import build_payloads_from_reference_tex
from .answer_render import compile_student_answer_pdfs
from .compile_tex import compile_tex_to_pdf
from .pdf_cleanse import CleanseReport, cleanse_test_pdf
from .reference_ranges import Key, find_reference_ranges
from .student_tex import parse_student_tex_answers
from .part_normalize import normalize_part


@dataclass(frozen=True)
class QABundleOutputs:
    """Outputs of bundle generation."""
    bundle_pdf: Path
    bundle_tex: Path
    student_clean_pdf: Path
    reference_clean_pdf: Path
    cleanse_report: CleanseReport
    ref_ranges: Dict[Key, Tuple[int, int]]
    student_ranges: Dict[Key, Tuple[int, int]]
    student_answer_pdfs: Dict[Key, Optional[Path]]


def _norm_key(k: Key) -> Key:
    q, p = k
    return (int(q), normalize_part(p))


def _union_keys(*dicts: Dict[Key, object]) -> list[Key]:
    s = set()
    for d in dicts:
        for k in d.keys():
            s.add(_norm_key(k))
    return sorted(s, key=lambda x: (x[0], x[1]))


def _tex_path(p: Path) -> str:
    return p.as_posix()


def _write_bundle_tex(
    out_tex: Path,
    *,
    title: str,
    font_name: str,
    reference_pdf: Optional[Path] = None,
    ref_ranges: Optional[Dict[Key, Tuple[int, int]]] = None,
    reference_snippets: Optional[Dict[Key, str]] = None,
    answer_pdfs: Dict[Key, Optional[Path]],
) -> None:
    """Single writer for both 'reference PDF pages' and 'reference TeX snippets' modes."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    if reference_pdf is None and reference_snippets is None:
        raise ValueError("Must provide either reference_pdf or reference_snippets.")

    ref_ranges = ref_ranges or {}
    reference_snippets = reference_snippets or {}

    keys = _union_keys(reference_snippets, ref_ranges, answer_pdfs)

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

    ref_pdf_tex = _tex_path(reference_pdf) if reference_pdf is not None else ""

    for (qnum, part) in keys:
        section_title = f"שאלה {qnum}" + (f"({part})" if part else "")
        tex.append(f"\\section{{{section_title}}}\n")

        tex.append("\\subsection*{Reference (question + official solution)}\n")
        if reference_pdf is not None:
            rng = ref_ranges.get((qnum, part)) or ref_ranges.get((qnum, ""))
            if rng:
                start, end = rng
                tex.append(f"\\includepdf[pages={{{start}-{end}}},pagecommand={{}}]{{{ref_pdf_tex}}}\n")
            else:
                tex.append("\\textcolor{red}{Missing reference page-range for this part.}\\par\n")
        else:
            ref_block = (reference_snippets.get((qnum, part)) or reference_snippets.get((qnum, "")) or "").strip()
            if not ref_block:
                tex.append("\\textcolor{red}{Missing reference snippet for this part.}\\par\n")
            else:
                tex.append("\\begingroup\n")
                tex.append(ref_block)
                tex.append("\n\\endgroup\n")

        tex.append("\\subsection*{Student answer (rendered)}\n")
        ap = answer_pdfs.get((qnum, part)) or answer_pdfs.get((qnum, ""))
        if ap is None:
            tex.append("\\textcolor{red}{Could not compile student answer for this part.}\\par\n")
        else:
            tex.append(f"\\includepdf[pages=-,pagecommand={{}}]{{{_tex_path(ap)}}}\n")

        tex.append("\\newpage\n")

    tex.append("\\end{document}\n")
    out_tex.write_text("".join(tex), encoding="utf-8")


def generate_bundle(
    *,
    reference: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    """Single entry point to generate a Q/A bundle PDF.

    `reference` can be either:
      - a .pdf with questions+official solution, or
      - a .tex/.txt with questions+official solution.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    suf = reference.suffix.lower()
    if suf == ".pdf":
        return _generate_from_reference_pdf(
            reference_pdf=reference,
            student_tex=student_tex,
            out_dir=out_dir,
            bundle_stem=bundle_stem,
            font_name=font_name,
        )
    if suf in (".tex", ".txt"):
        return _generate_from_reference_tex(
            reference_tex=reference,
            student_tex=student_tex,
            out_dir=out_dir,
            bundle_stem=bundle_stem,
            font_name=font_name,
        )

    raise ValueError(f"Unsupported reference type: {reference.name}. Expected .pdf or .tex/.txt")


def _generate_from_reference_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str,
    font_name: str,
) -> QABundleOutputs:
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    ref_ranges = { _norm_key(k): v for k, v in find_reference_ranges(reference_clean_pdf, out_dir).items() }
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed.\n"
            "Check build/debug_reference_pages.txt for extracted text.\n"
            "If it is empty, your PDF may be scanned (no selectable text)."
        )

    student_answers, student_ranges = parse_student_tex_answers(student_tex, out_dir)
    student_answers = { _norm_key(k): v for k, v in student_answers.items() }
    student_ranges = { _norm_key(k): v for k, v in student_ranges.items() }

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

    answer_pdfs = compile_student_answer_pdfs(student_answers, out_dir, font_name, clean=True)
    answer_pdfs = { _norm_key(k): v for k, v in answer_pdfs.items() }

    bundle_tex = out_dir / f"{bundle_stem}.tex"
    _write_bundle_tex(
        bundle_tex,
        title="Q/A Bundle (Reference pages + Rendered student answers)",
        font_name=font_name,
        reference_pdf=reference_clean_pdf,
        ref_ranges=ref_ranges,
        answer_pdfs=answer_pdfs,
    )

    bundle_pdf = compile_tex_to_pdf(
        bundle_tex,
        out_dir,
        clean=False,
        font_name=font_name,
        passes=2,
        texinputs=[bundle_tex.parent],
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

    return QABundleOutputs(
        bundle_pdf=bundle_pdf,
        bundle_tex=bundle_tex,
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
) -> QABundleOutputs:
    # 1) Build payloads in a dedicated folder (avoid clobbering ai_grade/payloads)
    payload_out = out_dir / "bundle_payloads"
    manifest_path, items = build_payloads_from_reference_tex(
        reference_tex=reference_tex,
        student_tex=student_tex,
        out_dir=payload_out,
        default_max_points=0.0,
    )

    # 2) Load payload JSONs -> canonical reference snippets + student answers
    reference_snippets: Dict[Key, str] = {}
    student_answers: Dict[Key, str] = {}
    student_ranges: Dict[Key, Tuple[int, int]] = {}  # keep empty unless you want to add ranges to payloads

    for it in items:
        data = json.loads(it.payload_path.read_text(encoding="utf-8"))
        k = _norm_key((data["key"]["qnum"], data["key"]["part"]))

        q = (data.get("reference", {}).get("question_text") or "").strip()
        sol = (data.get("reference", {}).get("solution_text") or "").strip()

        # Build the exact "reference (question+solution)" block the bundle uses
        ref_block = "\n\n".join([x for x in [q, sol] if x]).strip()
        reference_snippets[k] = ref_block

        s = (data.get("student", {}).get("latex_raw") or "").strip()
        student_answers[k] = s

    if not student_answers:
        raise RuntimeError("No student answers found in generated payloads (reference_tex path).")

    # 3) Optional: still compile whole student tex for debugging/inspection
    student_clean_pdf = compile_tex_to_pdf(
        student_tex,
        out_dir,
        clean=True,
        font_name=font_name,
        passes=2,
        texinputs=[student_tex.parent],
    ).pdf

    # 4) Render per-part student answer PDFs from the SAME student_latex used for grading
    answer_pdfs = compile_student_answer_pdfs(student_answers, out_dir, font_name)
    answer_pdfs = { _norm_key(k): v for k, v in answer_pdfs.items() }

    # 5) Write + compile the bundle as usual
    bundle_tex = out_dir / f"{bundle_stem}.tex"
    _write_bundle_tex(
        bundle_tex,
        title="Q/A Bundle (Reference TeX + Rendered student answers)",
        font_name=font_name,
        reference_snippets=reference_snippets,
        answer_pdfs=answer_pdfs,
    )

    bundle_pdf = compile_tex_to_pdf(
        bundle_tex,
        out_dir,
        clean=False,
        font_name=font_name,
        passes=2,
        texinputs=[bundle_tex.parent, reference_tex.parent, student_tex.parent],
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

    placeholder_pdf = out_dir / "reference_tex_placeholder.pdf"
    cleanse_report = CleanseReport(
        input_pdf=placeholder_pdf,
        output_pdf=placeholder_pdf,
        removed_pages_1based=(),
        kept_pages_1based=(),
        reason_lines=("Reference provided as .tex; PDF cleansing step skipped.",),
        debug_report_path=None,
    )

    return QABundleOutputs(
        bundle_pdf=bundle_pdf,
        bundle_tex=bundle_tex,
        student_clean_pdf=student_clean_pdf,
        reference_clean_pdf=placeholder_pdf,
        cleanse_report=cleanse_report,
        ref_ranges={},
        student_ranges=student_ranges,
        student_answer_pdfs=answer_pdfs,
    )

