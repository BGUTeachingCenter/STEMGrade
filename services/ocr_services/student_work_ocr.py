from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from services.file_handling.exam_structure import structure_to_student_tex_template
from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import StudentWorkOcrResult


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

_PART_RE = re.compile(
    r"^\s*(?:סעיף\s*)?[\(\[]?\s*([A-Za-zא-ת])\s*[\)\].:]\s*(.*)$",
    re.UNICODE,
)


def normalize_ocr_lines_for_student_parser(text: str) -> str:
    """
    Convert raw OCR text/MMD into markers supported by:
      services.file_handling.student_tex.parse_student_tex_answers

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
) -> StudentWorkOcrResult:
    """
    Task-specific student-work OCR processor.

    Input:
      OcrResponse from any provider.

    Output:
      StudentWorkOcrResult with reviewable student TeX.
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

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

    warnings: list[str] = []

    if not raw_text.strip():
        warnings.append("OCR returned empty text.")

    if "% WARNING: MathGrade could not confidently detect question titles." in student_tex:
        warnings.append("Could not confidently detect question titles from OCR text.")

    return StudentWorkOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        student_tex=student_tex,
        student_tex_path=student_tex_path,
        detected_question_count=None,
        detected_part_count=None,
        warnings=warnings,
        needs_teacher_review=True,
        ocr=ocr,
    )