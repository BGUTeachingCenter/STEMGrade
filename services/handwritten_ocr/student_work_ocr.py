from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from common.exam_structure import structure_to_student_tex_template
from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import StudentWorkOcrResult
from services.student_grading.student_answer_bundle import (
    parse_student_answer_bundle_from_model_text,
    student_answer_bundle_to_tex,
    write_student_answer_bundle,
)


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

_LATEX_HEADING_RE = re.compile(
    r"^\s*\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}\s*$",
    re.UNICODE,
)

_LEADING_NUMBER_IN_HEADING_RE = re.compile(
    r"^\s*([0-9]{1,2})(?:\s+|[).:\-–])+(.*)$",
    re.UNICODE,
)

_HEBREW_PART_WORD_RE = re.compile(
    r"^\s*(?:סעיף|פתרון\s+סעיף)\s+([א-תa-zA-Z])\s*[:.)-]?\s*(.*)$",
    re.UNICODE,
)

_Q_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:Question|Q|שאלה|תרגיל)\s*([0-9]{1,2})(?:\s*[\).:\-–]\s*([A-Za-zא-ת]))?.*$",
    re.IGNORECASE | re.UNICODE,
)

_Q_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?([0-9]{1,2})\s*[\).:]\s*$",
    re.UNICODE,
)

# Bare number at line start followed by a single part letter + delimiter:
#   "3. א."  →  Q3 part א
#   "6. א. בסיס האינדוקציה"  →  Q6 part א
# The part letter must be followed by a non-letter char (. ) : or space)
# to avoid matching word starts like "צ"ל" or "שטח".
_BARE_QNUM_PART_RE = re.compile(
    r"^\s*([0-9]{1,2})\s*[.)]\s+([א-תA-Za-z])\s*[.):]\s*(.*)",
    re.UNICODE,
)

# Bare number at line start followed by non-math content (no part letter):
#   "4. צ"ל:"  →  Q4
#   "5. שטח מלבן = ..."  →  Q5
_BARE_QNUM_TEXT_RE = re.compile(
    r"^\s*([0-9]{1,2})\s*[.)]\s+(\S.*)",
    re.UNICODE,
)

_PART_RE = re.compile(
    r"^\s*(?:סעיף\s*)?[\(\[]?\s*([A-Za-zא-ת])\s*[\)\].:]\s*(.*)$",
    re.UNICODE,
)


def normalize_ocr_lines_for_student_parser(text: str) -> str:
    """
    Convert raw OCR text/MMD into markers supported by:
      common.tex.student_tex.parse_student_tex_answers

    Parser-safe output:
      \\subsection*{Question 1}
      \\textbf{(א)}
      \\textbf{(ב)}

    This function is provider-neutral. It should not mention Mathpix/OpenAI.
    """
    raw_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_text = _MD_IMAGE_RE.sub("", raw_text)

    lines = raw_text.split("\n")

    out: list[str] = []
    seen_q: set[int] = set()
    detected_any_question = False

    def add_question(qnum: int) -> None:
        nonlocal detected_any_question

        if qnum < 1 or qnum > 99:
            return

        if qnum not in seen_q:
            out.append("")
            out.append(rf"\subsection*{{Question {qnum}}}")
            seen_q.add(qnum)
            detected_any_question = True

    def add_part(part: str, rest: str = "") -> None:
        part = (part or "").strip()
        rest = (rest or "").strip()

        if not part:
            return

        out.append(rf"\textbf{{({part})}}")

        if rest:
            out.append(rest)

    for raw in lines:
        line = raw.strip()

        if not line:
            out.append("")
            continue

        line = _MD_IMAGE_RE.sub("", line).strip()
        if not line:
            continue

        heading_match = _LATEX_HEADING_RE.match(line)
        if heading_match:
            heading_text = heading_match.group(1).strip()

            q_match = _Q_RE.match(heading_text)
            if q_match:
                qnum = int(q_match.group(1))
                add_question(qnum)

                if q_match.group(2):
                    add_part(q_match.group(2))

                continue

            leading_num = _LEADING_NUMBER_IN_HEADING_RE.match(heading_text)
            if leading_num:
                qnum = int(leading_num.group(1))
                add_question(qnum)
                continue

            out.append(rf"\textbf{{{heading_text}}}")
            continue

        q_match = _Q_RE.match(line)
        if q_match:
            qnum = int(q_match.group(1))
            add_question(qnum)

            if q_match.group(2):
                add_part(q_match.group(2))

            continue

        q_line_match = _Q_LINE_RE.match(line)
        if q_line_match:
            qnum = int(q_line_match.group(1))
            add_question(qnum)
            continue

        bare_qp = _BARE_QNUM_PART_RE.match(line)
        if bare_qp:
            qnum = int(bare_qp.group(1))
            add_question(qnum)
            add_part(bare_qp.group(2), bare_qp.group(3))
            continue

        bare_qt = _BARE_QNUM_TEXT_RE.match(line)
        if bare_qt:
            qnum = int(bare_qt.group(1))
            if qnum > max(seen_q, default=0):
                add_question(qnum)
                out.append(bare_qt.group(2))
                continue

        heb_part_match = _HEBREW_PART_WORD_RE.match(line)
        if heb_part_match:
            add_part(heb_part_match.group(1), heb_part_match.group(2))
            continue

        part_match = _PART_RE.match(line)
        if part_match:
            add_part(part_match.group(1), part_match.group(2))
            continue

        out.append(line)

    body = "\n".join(out).strip()

    if not detected_any_question:
        body = "\n".join(
            [
                r"% WARNING: MathGrade could not confidently detect question titles.",
                r"% Before grading, split this OCR text into:",
                r"% \subsection*{Question 1}",
                r"% \textbf{(א)}",
                r"% \textbf{(ב)}",
                "",
                r"\subsection*{Question 1}",
                "",
                body,
            ]
        ).strip()

    return body


def ocr_text_to_student_tex(
    text: str,
    *,
    source_name: str = "",
    exam_structure: dict[str, Any] | None = None,
) -> str:
    """
    Convert provider-neutral OCR text into reviewable student-answer TeX.

    This produces the format expected by parse_student_tex_answers():
      \\subsection*{Question N}
      \\textbf{(א)}
    """
    if exam_structure:
        skeleton = structure_to_student_tex_template(
            exam_structure,
            placeholder="% Paste / review OCR answer here.",
        )

        body = "\n".join(
            [
                skeleton,
                "",
                r"\section*{Raw OCR text for review}",
                r"% Copy answers from this raw OCR section into the matching question/part above.",
                "",
                normalize_ocr_lines_for_student_parser(text),
            ]
        ).strip()
    else:
        body = normalize_ocr_lines_for_student_parser(text)

    return "\n".join(
        [
            r"\documentclass[12pt]{article}",
            r"\usepackage[a4paper,margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,mathtools}",
            r"\usepackage{fontspec}",
            r"\usepackage{polyglossia}",
            r"\setmainlanguage{hebrew}",
            r"\setotherlanguage{english}",
            r"\newfontfamily\hebrewfont{Arial}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.6em}",
            "",
            r"\begin{document}",
            r"% OCR draft generated by MathGrade.",
            rf"% Source: {source_name}" if source_name else r"% Source: OCR upload",
            r"% Review carefully before grading.",
            r"% Required grading markers:",
            r"%   \subsection*{Question 1}",
            r"%   \textbf{(א)}",
            r"%   \textbf{(ב)}",
            "",
            body,
            "",
            r"\end{document}",
            "",
        ]
    )


def build_student_work_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str = "",
    source_name: str = "",
    out_dir: Path | None = None,
    exam_structure: dict[str, Any] | None = None,
    document_type: str = "unknown",
    ocr_path: str = "generic_ocr",
) -> StudentWorkOcrResult:
    """
    Task-specific student-work OCR processor.

    Input:
      OcrResponse from any provider, plus the OCR routing decision
      (document_type / ocr_path) so the result records how it was produced.

    Output:
      StudentWorkOcrResult with reviewable student TeX. When the OCR used the
      structured "student_answer_bundle" prompt (vision providers on handwritten
      scans), the primary text is JSON; we parse it into a normalized bundle and
      render the TeX from that. Otherwise we fall back to the text->TeX path.
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    warnings: list[str] = []
    student_answer_bundle: dict[str, Any] = {}
    student_answer_bundle_path = ""

    # Returns None when the text is not a usable JSON answer bundle, so this is
    # safe to attempt for every path (plain OCR text simply falls through).
    bundle = parse_student_answer_bundle_from_model_text(
        raw_text,
        source_name=source_name,
        provider=ocr.provider or "",
        model=ocr.model,
        document_type=document_type,
    )

    if bundle is not None:
        student_answer_bundle = bundle
        student_tex = student_answer_bundle_to_tex(bundle, source_name=source_name)
        warnings.extend(bundle.get("warnings") or [])
    else:
        student_tex = ocr_text_to_student_tex(
            raw_text,
            source_name=source_name,
            exam_structure=exam_structure,
        )

    student_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "ocr_student_answer.tex"
        path.write_text(student_tex, encoding="utf-8")
        student_tex_path = str(path)

        if student_answer_bundle:
            student_answer_bundle_path = str(
                write_student_answer_bundle(student_answer_bundle, out_dir)
            )

    if not raw_text.strip():
        warnings.append("OCR returned empty text.")

    if "% WARNING: MathGrade could not confidently detect question titles." in student_tex:
        warnings.append("Could not confidently detect question titles from OCR text.")

    detected_question_count: int | None = None
    detected_part_count: int | None = None
    if student_answer_bundle:
        answers = student_answer_bundle.get("answers") or []
        detected_part_count = len(answers)
        detected_question_count = len(
            {a.get("question_id") for a in answers if a.get("question_id") is not None}
        )

    return StudentWorkOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        student_tex=student_tex,
        student_tex_path=student_tex_path,
        student_answer_bundle=student_answer_bundle,
        student_answer_bundle_path=student_answer_bundle_path,
        document_type=document_type,
        ocr_path=ocr_path,
        detected_question_count=detected_question_count,
        detected_part_count=detected_part_count,
        warnings=warnings,
        needs_teacher_review=True,
        ocr=ocr,
    )