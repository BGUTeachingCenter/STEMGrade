from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common.tex.latex_render import latex_render_mixed
from common.tex.math_normalize import normalize_math_text
from common.tex.part_normalize import normalize_part
from services.student_grading.student_answer_bundle import (
    normalize_student_answer_bundle,
)
from typing import Dict, List, Tuple


# Match \section{...} and \section*{...}
_SECTION_RE = re.compile(
    r"\\section\*?\{(?P<title>[^}]*)\}(?P<body>.*?)(?=(\\section\*?\{)|\\end\{document\})",
    re.DOTALL,
)

# Match Q1 or Q1(a)
_QID_RE = re.compile(r"Q\s*(\d+)(?:\(([^)]+)\))?", re.IGNORECASE)

# Inline QA bundle headings like: {\large \textbf{Q1(a)}}\par
_INLINE_QA_HEAD_RE = re.compile(
    r"\{\s*\\large\s+\\textbf\{(?P<label>Q\s*\d+(?:\([^)]+\))?)\}\s*\}\s*\\par",
    re.IGNORECASE,
)

# Inline feedback headings like: {\Large Q1(a)\par}
_INLINE_FB_HEAD_RE = re.compile(
    r"\{\s*\\Large\s+(?P<label>Q\s*\d+(?:\([^)]+\))?)\s*\\par\s*\}",
    re.IGNORECASE,
)

# Markers in inline QA bundle
_INLINE_REF_MARK = r"{\bfseries Reference}\par"
_INLINE_STU_MARK = r"{\bfseries Student answer}\par"


def build_unified_tex(
    *,
    qa_tex: Path,
    feedback_tex: Path,
    out_dir: Path,
    output_stem: str = "graded_union",
) -> Path:
    """
    Build one final TeX file interleaved by question:

      Q:
        1) Student answer
        2) Reference
        3) Feedback

    Supports inputs in TWO styles:
      A) Section style:
         \\section{Q1...} ... \\subsection*{Reference} ... \\subsection*{Student answer} ...
      B) Inline style (your current qa_bundle.tex / feedback.tex):
         \\hrule
         {\\large \\textbf{Q1(a)}}\\par
         {\\bfseries Reference}\\par
         ...
         {\\bfseries Student answer}\\par
         ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_text = qa_tex.read_text(encoding="utf-8", errors="replace")
    fb_text = feedback_tex.read_text(encoding="utf-8", errors="replace")

    qa_body = _extract_document_body(qa_text)
    fb_body = _extract_document_body(fb_text)

    qa_sections = _parse_qa_any(qa_body)
    fb_sections = _parse_feedback_any(fb_body)

    ordered_keys = _ordered_union_keys(qa_sections, fb_sections)

    unified_parts: List[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{enumitem}",
        r"\usepackage{fontspec}",
        r"\setmainfont{Arial}",
        r"\usepackage{bidi}",
        r"\setRTL",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\begin{document}",
    ]

    if not ordered_keys:
        unified_parts.extend([
            r"\section{Unification failed}",
            r"\textcolor{red}{No question blocks (QIDs) could be parsed from qa\_tex and feedback\_tex.}",
            r"\par",
            r"\textbf{This usually means the parser does not match the bundle/feedback format.}",
            r"\par\medskip",
            r"\textbf{QA body starts with:}\par",
            r"\begin{verbatim}",
            _truncate_for_verbatim(qa_body, 1200),
            r"\end{verbatim}",
            r"\textbf{Feedback body starts with:}\par",
            r"\begin{verbatim}",
            _truncate_for_verbatim(fb_body, 1200),
            r"\end{verbatim}",
            r"\end{document}",
            "",
        ])
        out_path = out_dir / f"{output_stem}.tex"
        out_path.write_text("\n".join(unified_parts), encoding="utf-8")
        return out_path

    for qid in ordered_keys:
        qa = qa_sections.get(qid, {})
        fb = (fb_sections.get(qid) or "").strip()

        section_title = qa.get("title") or qid
        student_block = (qa.get("student") or "").strip()
        reference_block = (qa.get("reference") or "").strip()

        unified_parts.extend([
            f"\\section{{{_tex_escape_title(section_title)}}}",
            r"\subsection*{Student answer}",
            student_block if student_block else r"\textcolor{red}{Missing student answer block.}",
            r"\subsection*{Reference}",
            reference_block if reference_block else r"\textcolor{red}{Missing reference block.}",
            r"\subsection*{Feedback}",
            fb if fb else r"\textcolor{red}{Missing feedback block.}",
            r"\clearpage",
        ])

    unified_parts.append(r"\end{document}")
    unified_parts.append("")

    out_path = out_dir / f"{output_stem}.tex"
    out_path.write_text("\n".join(unified_parts), encoding="utf-8")
    return out_path


def _extract_document_body(tex: str) -> str:
    start_token = r"\begin{document}"
    end_token = r"\end{document}"

    start = tex.find(start_token)
    end = tex.rfind(end_token)

    if start != -1 and end != -1 and end > start:
        return tex[start + len(start_token):end].strip()

    return tex.strip()


# -------------------------
# Parsing QA (either section style or inline style)
# -------------------------

def _parse_qa_any(body: str) -> Dict[str, Dict[str, str]]:
    # Try section-style first
    sections = _parse_qa_sections_section_style(body)
    if sections:
        return sections
    # Fallback: inline-style (qa_bundle.tex)
    return _parse_qa_sections_inline_style(body)


def _parse_qa_sections_section_style(body: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}

    for match in _SECTION_RE.finditer(body):
        title = (match.group("title") or "").strip()
        section_body = (match.group("body") or "").strip()

        qid = _qid_from_title(title)
        if not qid:
            continue

        reference_block = _extract_subsection_block(section_body, "Reference")
        student_block = _extract_subsection_block(section_body, "Student answer")

        out[qid] = {
            "title": title,
            "student": student_block.strip(),
            "reference": reference_block.strip(),
        }

    return out


def _parse_qa_sections_inline_style(body: str) -> Dict[str, Dict[str, str]]:
    r"""
    Parse inline Q/A bundle:

      \hrule
      {\large \textbf{Q1(a)}}\par
      ...
      {\bfseries Reference}\par
      <reference...>
      ...
      {\bfseries Student answer}\par
      <student...>
      ...
      \hrule (next) OR end
    """
    out: Dict[str, Dict[str, str]] = {}

    matches = list(_INLINE_QA_HEAD_RE.finditer(body))
    if not matches:
        return out

    for i, m in enumerate(matches):
        label = (m.group("label") or "").strip()
        qid = _normalize_qid(label)
        if not qid:
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end]

        ref_block, stu_block = _extract_inline_ref_and_student(chunk)

        out[qid] = {
            "title": qid,
            "reference": ref_block.strip(),
            "student": stu_block.strip(),
        }

    return out


def _extract_inline_ref_and_student(chunk: str) -> Tuple[str, str]:
    r"""
    Extract reference and student blocks from the inline chunk using the markers:
      {\bfseries Reference}\par
      {\bfseries Student answer}\par
    """
    idx_ref = chunk.find(_INLINE_REF_MARK)
    idx_stu = chunk.find(_INLINE_STU_MARK)

    if idx_ref == -1 and idx_stu == -1:
        # Nothing recognized; return entire chunk as reference to avoid losing data.
        return chunk.strip(), ""

    if idx_ref != -1 and idx_stu != -1 and idx_stu > idx_ref:
        ref_start = idx_ref + len(_INLINE_REF_MARK)
        ref_text = chunk[ref_start:idx_stu].strip()

        stu_start = idx_stu + len(_INLINE_STU_MARK)
        stu_text = chunk[stu_start:].strip()
        return ref_text, stu_text

    # If order is weird, do best effort:
    if idx_ref != -1:
        ref_start = idx_ref + len(_INLINE_REF_MARK)
        return chunk[ref_start:].strip(), ""
    if idx_stu != -1:
        stu_start = idx_stu + len(_INLINE_STU_MARK)
        return "", chunk[stu_start:].strip()

    return "", ""


# -------------------------
# Parsing Feedback (either section style or inline style)
# -------------------------

def _parse_feedback_any(body: str) -> Dict[str, str]:
    # Try section-style first (rare in your current feedback.tex)
    out = _parse_feedback_sections_section_style(body)
    if out:
        return out
    # Fallback: inline-style feedback
    return _parse_feedback_sections_inline_style(body)


def _parse_feedback_sections_section_style(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for match in _SECTION_RE.finditer(body):
        title = (match.group("title") or "").strip()
        qid = _qid_from_title(title)
        if not qid:
            continue
        out[qid] = (match.group("body") or "").strip()
    return out


def _parse_feedback_sections_inline_style(body: str) -> Dict[str, str]:
    r"""
    Parse inline feedback:

      \hrule ...
      {\Large Q1(a)\par}
      <feedback text ...>
      \hrule (next) OR end
    """
    out: Dict[str, str] = {}

    matches = list(_INLINE_FB_HEAD_RE.finditer(body))
    if matches:
        for i, m in enumerate(matches):
            label = (m.group("label") or "").strip()
            qid = _normalize_qid(label)
            if not qid:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            chunk = body[start:end].strip()
            out[qid] = chunk
        return out

    # Secondary fallback: search for plain "Q1(a)" occurrences at line starts
    qid_heading_re = re.compile(
        r"(?P<head>(?:^|\n)\s*(Q\s*\d+(?:\([^)]+\))?))",
        re.IGNORECASE,
    )
    matches2 = list(qid_heading_re.finditer(body))
    if not matches2:
        return out

    for i, m in enumerate(matches2):
        qid = _normalize_qid(m.group(1))
        if not qid:
            continue
        start = m.start()
        end = matches2[i + 1].start() if i + 1 < len(matches2) else len(body)
        out[qid] = body[start:end].strip()

    return out


# -------------------------
# Helpers
# -------------------------

def _extract_subsection_block(section_body: str, subsection_name: str) -> str:
    pattern = re.compile(
        rf"\\subsection\*\{{{re.escape(subsection_name)}\}}(?P<body>.*?)(?=(\\subsection\*|\\section\*?\{{|$))",
        re.DOTALL,
    )
    match = pattern.search(section_body)
    return (match.group("body") if match else "").strip()


def _qid_from_title(title: str) -> str:
    title = (title or "").strip()

    qid = _normalize_qid(title)
    if qid:
        return qid

    heb_match = re.search(r"שאלה\s*(\d+)(?:\(([^)]+)\))?", title)
    if heb_match:
        qnum = heb_match.group(1)
        part = (heb_match.group(2) or "").strip()
        return f"Q{qnum}({part})" if part else f"Q{qnum}"

    return ""


def _normalize_qid(text: str) -> str:
    match = _QID_RE.search(text or "")
    if not match:
        return ""
    qnum = match.group(1)
    part = (match.group(2) or "").strip()
    return f"Q{qnum}({part})" if part else f"Q{qnum}"


def _ordered_union_keys(
    qa_sections: Dict[str, Dict[str, str]],
    feedback_sections: Dict[str, str],
) -> List[str]:
    keys = set(qa_sections.keys()) | set(feedback_sections.keys())

    def sort_key(qid: str) -> Tuple[int, str]:
        m = _QID_RE.search(qid)
        if not m:
            return (10**9, qid)
        qnum = int(m.group(1))
        part = (m.group(2) or "").strip()
        return (qnum, part)

    return sorted(keys, key=sort_key)


def _truncate_for_verbatim(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n...\n(TRUNCATED)\n"


def _tex_escape_title(text: str) -> str:
    # safe-ish escaping for section titles
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
    )


# =========================================================
# Canonical JSON-driven graded result
# =========================================================


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def _positive_number(
    value: Any,
) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None

    return (
        number
        if number > 0
        else None
    )


def _looks_like_orphan_question_fragment(
    line: str,
) -> bool:
    """
    Detect parser debris such as:

      ^n k^2=...
      {b}<\\frac{c}{d}\\)}
      +\\frac1{n+2}+...

    These fragments appeared after a clean Hebrew question line when the
    source question JSON lost the beginning of a math expression.
    """
    text = str(
        line or ""
    ).strip()

    if not text:
        return False

    if re.search(
        r"[\u0590-\u05FF]",
        text,
    ):
        return False

    if text.startswith(
        (
            r"\[",
            r"\(",
            "$",
        )
    ):
        return False

    if text.startswith(
        (
            "^",
            "+",
        )
    ):
        return True

    if text.startswith("{") and any(
        token in text
        for token in (
            r"\frac",
            r"\right",
            r"\binom",
            r"\)",
        )
    ):
        return True

    return False


def _clean_question_text_for_student(
    value: Any,
) -> str:
    """
    Remove payload labels and obvious incomplete math fragments from the
    student-facing question text.

    The internal graded result remains untouched for debugging.
    """
    text = (
        str(value or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not text:
        return ""

    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line in {
            "Question:",
            "Required action:",
        }:
            continue

        if line.startswith("Question:"):
            line = line[
                len("Question:"):
            ].strip()

        if line.startswith(
            "Required action:"
        ):
            line = line[
                len("Required action:"):
            ].strip()

        if (
            _looks_like_orphan_question_fragment(
                line
            )
        ):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(
        cleaned_lines
    )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    ).strip()

    return cleaned


def build_student_feedback_json(
    *,
    graded_result_json: Path,
    out_dir: Path,
    student_answer_bundle_json: Path | None = None,
    output_name: str = (
        "student_feedback.json"
    ),
) -> Path:
    """
    Create the JSON that is safe to expose to students.

    It intentionally excludes:
      - official/reference solutions
      - mismatch targets
      - confidence values
      - internal error taxonomy
      - provider evidence/debug fields
    """
    data = json.loads(
        Path(
            graded_result_json
        ).read_text(
            encoding="utf-8"
        )
    )

    answer_locations: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    if student_answer_bundle_json:
        bundle_path = Path(
            student_answer_bundle_json
        )

        if (
            bundle_path.exists()
            and bundle_path.suffix.lower()
            == ".json"
        ):
            try:
                raw_bundle = json.loads(
                    bundle_path.read_text(
                        encoding="utf-8"
                    )
                )

                normalized_bundle = (
                    normalize_student_answer_bundle(
                        raw_bundle,
                        source_name=(
                            bundle_path.name
                        ),
                    )
                )

                for answer in (
                    normalized_bundle.get(
                        "answers"
                    )
                    or []
                ):
                    try:
                        answer_question_id = int(
                            answer.get(
                                "question_id"
                            )
                        )
                    except Exception:
                        continue

                    answer_part_key = (
                        normalize_part(
                            str(
                                answer.get(
                                    "part_key"
                                )
                                or ""
                            )
                        )
                    )

                    answer_locations[
                        (
                            answer_question_id,
                            answer_part_key,
                        )
                    ] = {
                        "page_numbers": list(
                            answer.get(
                                "page_numbers"
                            )
                            or []
                        ),
                        "regions": list(
                            answer.get(
                                "regions"
                            )
                            or []
                        ),
                    }
            except Exception:
                answer_locations = {}

    raw_parts = (
        data.get("parts")
        if isinstance(
            data.get("parts"),
            list,
        )
        else []
    )

    student_parts: list[
        dict[str, Any]
    ] = []

    for part in raw_parts:
        if not isinstance(
            part,
            dict,
        ):
            continue

        part_scoring_mode = str(
            part.get("scoring_mode")
            or "feedback_only"
        ).strip()

        student_answer = str(
            part.get(
                "student_answer"
            )
            or ""
        ).strip()

        correctness_level = str(
            part.get(
                "correctness_level"
            )
            or (
                "not_answered"
                if not student_answer
                else "needs_work"
            )
        ).strip().lower()

        if correctness_level not in {
            "correct",
            "mostly_correct",
            "partially_correct",
            "needs_work",
            "not_answered",
        }:
            correctness_level = (
                "not_answered"
                if not student_answer
                else "needs_work"
            )

        try:
            question_id = int(
                part.get(
                    "question_id"
                )
            )
        except Exception:
            question_id = None

        part_key = normalize_part(
            str(
                part.get(
                    "part_key"
                )
                or ""
            )
        )

        location = (
            answer_locations.get(
                (
                    question_id,
                    part_key,
                ),
                {},
            )
            if question_id is not None
            else {}
        )

        feedback = {
            "summary": str(
                part.get("summary")
                or ""
            ).strip(),
            "what_was_correct": (
                _as_string_list(
                    part.get(
                        "what_was_correct"
                    )
                )
            ),
            "main_mistakes": (
                _as_string_list(
                    part.get(
                        "main_mistakes"
                    )
                )
            ),
            "how_to_improve": (
                _as_string_list(
                    part.get(
                        "how_to_improve"
                    )
                )
            ),
            "suggested_next_step": str(
                part.get(
                    "suggested_next_step"
                )
                or ""
            ).strip(),
        }

        student_parts.append(
            {
                "qid": str(
                    part.get("qid")
                    or ""
                ).strip(),
                "question_id": (
                    question_id
                ),
                "part_key": part_key,
                "question_text": (
                    _clean_question_text_for_student(
                        part.get(
                            "question_text"
                        )
                    )
                ),
                "student_answer": (
                    student_answer
                ),
                "page_numbers": list(
                    location.get(
                        "page_numbers"
                    )
                    or []
                ),
                "regions": list(
                    location.get(
                        "regions"
                    )
                    or []
                ),
                "correctness": {
                    "level": (
                        correctness_level
                    ),
                },
                "scoring_mode": (
                    part_scoring_mode
                ),
                "score": (
                    part.get("score")
                    if part_scoring_mode
                    == "scored"
                    else None
                ),
                "max_score": (
                    part.get(
                        "max_score"
                    )
                    if part_scoring_mode
                    == "scored"
                    else None
                ),
                "feedback": feedback,
            }
        )

    scoring = (
        data.get("scoring")
        if isinstance(
            data.get("scoring"),
            dict,
        )
        else {}
    )

    score_available = bool(
        scoring.get(
            "score_available"
        )
    )

    result = {
        "schema_version": (
            "student_feedback_v1"
        ),
        "created_at": str(
            data.get("created_at")
            or ""
        ),
        "exam_id": str(
            data.get("exam_id")
            or ""
        ),
        "provider": str(
            data.get("provider")
            or ""
        ),
        "subject": str(
            data.get("subject")
            or "math"
        ),
        "source_filename": str(
            data.get(
                "source_filename"
            )
            or ""
        ),
        "visual_overlay": {
            "schema_version": (
                "student_feedback_overlay_v1"
            ),
            "coordinate_space": (
                "normalized_page"
            ),
            "has_exact_coordinates": any(
                bool(
                    part.get(
                        "regions"
                    )
                )
                for part in student_parts
            ),
        },
        "scoring": {
            "mode": str(
                scoring.get("mode")
                or "feedback_only"
            ),
            "score_available": (
                score_available
            ),
            "total_score": (
                data.get(
                    "total_score"
                )
                if score_available
                else None
            ),
            "total_max_score": (
                data.get(
                    "total_max_score"
                )
                if score_available
                else None
            ),
        },
        "parts": student_parts,
    }

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / output_name
    )

    out_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out_path


def _payload_qid(
    payload: dict[str, Any],
    manifest_item: dict[str, Any],
    index: int,
) -> str:
    raw_qid = str(
        payload.get("qid")
        or manifest_item.get("qid")
        or payload.get("question_id")
        or ""
    ).strip()

    normalized = _normalize_qid(raw_qid)
    if normalized:
        return normalized

    key = (
        payload.get("key")
        if isinstance(payload.get("key"), dict)
        else {}
    )

    qnum = (
        key.get("qnum")
        or key.get("question_id")
        or index
    )

    part = str(
        key.get("part")
        or key.get("part_key")
        or ""
    ).strip()

    try:
        qnum_text = str(int(qnum))
    except Exception:
        qnum_text = (
            str(qnum or index).strip()
            or str(index)
        )

    if part:
        return f"Q{qnum_text}({part})"

    return f"Q{qnum_text}"


def _qid_components(
    qid: str,
    payload: dict[str, Any],
) -> tuple[int | None, str]:
    key = (
        payload.get("key")
        if isinstance(payload.get("key"), dict)
        else {}
    )

    qnum_raw = (
        key.get("qnum")
        or key.get("question_id")
    )

    part = str(
        key.get("part")
        or key.get("part_key")
        or ""
    ).strip()

    match = _QID_RE.search(qid or "")

    if match:
        if qnum_raw in (None, ""):
            qnum_raw = match.group(1)

        if not part:
            part = str(
                match.group(2) or ""
            ).strip()

    try:
        qnum = int(qnum_raw)
    except Exception:
        qnum = None

    return qnum, part


def build_graded_result_json(
    *,
    manifest_json: Path,
    grades_json: Path,
    out_dir: Path,
    exam_id: str,
    provider: str,
    student_code: str = "",
    source_filename: str = "",
    output_name: str = "graded_result.json",
) -> Path:
    """
    Merge grading payloads and grades.json into one canonical result.

    The payload supplies:
      - question text
      - student answer
      - reference solution
      - part identity

    grades.json supplies:
      - score
      - summary
      - correct elements
      - mistakes
      - improvement advice
    """
    manifest_json = Path(manifest_json)
    grades_json = Path(grades_json)

    manifest = json.loads(
        manifest_json.read_text(
            encoding="utf-8"
        )
    )

    grades = json.loads(
        grades_json.read_text(
            encoding="utf-8"
        )
    )

    manifest_items = (
        manifest.get("items")
        if isinstance(
            manifest.get("items"),
            list,
        )
        else []
    )

    grade_items = (
        grades.get("question_grades")
        or grades.get("questions")
        or []
    )

    if not isinstance(grade_items, list):
        grade_items = []

    # Grade entries are stored in buckets because duplicate QIDs should
    # not silently overwrite one another.

    grade_buckets: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for grade in grade_items:
        if not isinstance(grade, dict):
            continue

        raw_grade_qid = str(
            grade.get("qid")
            or grade.get("id")
            or grade.get("question_id")
            or ""
        ).strip()

        normalized_grade_qid = (
            _normalize_qid(raw_grade_qid)
        )

        grade_key = (
            normalized_grade_qid
            or raw_grade_qid
        ).lower()

        grade_buckets.setdefault(
            grade_key,
            [],
        ).append(grade)

    payload_dir = (
        manifest_json.parent
        / "payloads"
    )

    parts: list[dict[str, Any]] = []

    for index, manifest_item in enumerate(
        manifest_items,
        start=1,
    ):
        if not isinstance(
            manifest_item,
            dict,
        ):
            continue

        payload_filename = str(
            manifest_item.get(
                "payload_file"
            )
            or ""
        ).strip()

        if not payload_filename:
            continue

        payload_path = (
            payload_dir
            / payload_filename
        )

        if not payload_path.exists():
            continue

        payload = json.loads(
            payload_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, dict):
            continue

        qid = _payload_qid(
            payload,
            manifest_item,
            index,
        )

        question_id, part_key = (
            _qid_components(
                qid,
                payload,
            )
        )

        grade_key = (
            _normalize_qid(qid)
            or qid
        ).lower()

        grade_bucket = (
            grade_buckets.get(grade_key)
            or []
        )

        grade = (
            grade_bucket.pop(0)
            if grade_bucket
            else {}
        )

        reference = (
            payload.get("reference")
            if isinstance(
                payload.get("reference"),
                dict,
            )
            else {}
        )

        student = (
            payload.get("student")
            if isinstance(
                payload.get("student"),
                dict,
            )
            else {}
        )

        rubric = (
            payload.get("rubric")
            if isinstance(
                payload.get("rubric"),
                dict,
            )
            else {}
        )

        max_score = grade.get(
            "max_points"
        )

        if max_score in (None, ""):
            max_score = (
                payload.get("max_points")
                or rubric.get("score_max")
                or 0
            )

        student_answer = str(
            student.get("latex_raw")
            or student.get("latex_clean")
            or ""
        ).strip()

        correctness_level = str(
            grade.get(
                "correctness_level"
            )
            or (
                "not_answered"
                if not student_answer
                else "needs_work"
            )
        ).strip().lower()

        allowed_correctness_levels = {
            "correct",
            "mostly_correct",
            "partially_correct",
            "needs_work",
            "not_answered",
        }

        if (
                correctness_level
                not in allowed_correctness_levels
        ):
            correctness_level = (
                "not_answered"
                if not student_answer
                else "needs_work"
            )

        part_scoring_mode = str(
            grade.get("scoring_mode")
            or (
                "scored"
                if _positive_number(
                    max_score
                )
                else "feedback_only"
            )
        ).strip()

        part_score = (
            grade.get("score")
            if part_scoring_mode
               == "scored"
            else None
        )

        part_max_score = (
            max_score
            if part_scoring_mode
               == "scored"
            else None
        )

        parts.append(
            {
                "qid": qid,
                "question_id": question_id,
                "part_key": part_key,
                "question_text": str(
                    reference.get(
                        "question_text"
                    )
                    or ""
                ).strip(),
                "student_answer": (
                    student_answer
                ),
                "reference_solution": str(
                    reference.get(
                        "solution_text"
                    )
                    or ""
                ).strip(),
                "correctness_level": (
                    correctness_level
                ),
                "scoring_mode": (
                    part_scoring_mode
                ),
                "score": part_score,
                "max_score": (
                    part_max_score
                ),
                "summary": str(
                    grade.get("summary")
                    or ""
                ).strip(),
                "what_was_correct": (
                    _as_string_list(
                        grade.get(
                            "what_was_correct"
                        )
                    )
                ),
                "main_mistakes": (
                    _as_string_list(
                        grade.get(
                            "main_mistakes"
                        )
                    )
                ),
                "how_to_improve": (
                    _as_string_list(
                        grade.get(
                            "how_to_improve"
                        )
                    )
                ),
                "suggested_next_step": str(
                    grade.get(
                        "suggested_next_step_he"
                    )
                    or grade.get(
                        "suggested_next_step"
                    )
                    or ""
                ).strip(),
                "confidence": grade.get(
                    "confidence"
                ),
                "mismatch": (
                    grade.get("mismatch")
                    if isinstance(
                        grade.get("mismatch"),
                        dict,
                    )
                    else {}
                ),
                "common_errors": (
                    _as_string_list(
                        grade.get(
                            "common_errors_detected"
                        )
                    )
                ),
                "ocr_missing": (
                    not bool(student_answer)
                ),
            }
        )

    # Preserve any grade that did not match a payload.
    # This makes QID mismatches visible instead of silently losing feedback.
    for remaining_bucket in (
        grade_buckets.values()
    ):
        for grade in remaining_bucket:
            raw_qid = str(
                grade.get("qid")
                or grade.get("id")
                or "Q?"
            ).strip()

            qid = (
                _normalize_qid(raw_qid)
                or raw_qid
            )

            question_id, part_key = (
                _qid_components(
                    qid,
                    {},
                )
            )

            parts.append(
                {
                    "qid": qid,
                    "question_id": (
                        question_id
                    ),
                    "part_key": part_key,
                    "question_text": "",
                    "student_answer": "",
                    "reference_solution": "",
                    "correctness_level": str(
                        grade.get(
                            "correctness_level"
                        )
                        or "not_answered"
                    ).strip().lower(),
                    "scoring_mode": (
                        str(
                            grade.get(
                                "scoring_mode"
                            )
                            or (
                                "scored"
                                if _positive_number(
                                    grade.get(
                                        "max_points"
                                    )
                                )
                                else (
                                    "feedback_only"
                                )
                            )
                        )
                    ),
                    "score": (
                        grade.get("score")
                        if _positive_number(
                            grade.get(
                                "max_points"
                            )
                        )
                        else None
                    ),
                    "max_score": (
                        grade.get(
                            "max_points"
                        )
                        if _positive_number(
                            grade.get(
                                "max_points"
                            )
                        )
                        else None
                    ),
                    "summary": str(
                        grade.get("summary")
                        or ""
                    ).strip(),
                    "what_was_correct": (
                        _as_string_list(
                            grade.get(
                                "what_was_correct"
                            )
                        )
                    ),
                    "main_mistakes": (
                        _as_string_list(
                            grade.get(
                                "main_mistakes"
                            )
                        )
                    ),
                    "how_to_improve": (
                        _as_string_list(
                            grade.get(
                                "how_to_improve"
                            )
                        )
                    ),
                    "suggested_next_step": str(
                        grade.get(
                            "suggested_next_step_he"
                        )
                        or ""
                    ).strip(),
                    "confidence": grade.get(
                        "confidence"
                    ),
                    "mismatch": (
                        grade.get("mismatch")
                        if isinstance(
                            grade.get(
                                "mismatch"
                            ),
                            dict,
                        )
                        else {}
                    ),
                    "common_errors": (
                        _as_string_list(
                            grade.get(
                                "common_errors_detected"
                            )
                        )
                    ),
                    "ocr_missing": True,
                }
            )

    scored_parts = [
        part
        for part in parts
        if isinstance(
            part,
            dict,
        )
           and part.get(
            "scoring_mode"
        ) == "scored"
    ]

    if (
            parts
            and len(scored_parts)
            == len(parts)
    ):
        scoring_mode = "scored"

    elif scored_parts:
        scoring_mode = "mixed"

    else:
        scoring_mode = (
            "feedback_only"
        )

    score_available = bool(
        scored_parts
    )

    result = {
        "schema_version": (
            "graded_result_v1"
        ),
        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "exam_id": str(
            exam_id
            or manifest.get("exam_id")
            or ""
        ),
        "provider": str(
            provider or ""
        ),
        "student_code": str(
            student_code or ""
        ),
        "source_filename": str(
            source_filename or ""
        ),
        "subject": str(
            manifest.get("subject")
            or "math"
        ),
                "scoring": {
            "mode": scoring_mode,
            "score_available": (
                score_available
            ),
            "scored_part_count": len(
                scored_parts
            ),
            "feedback_only_part_count": (
                len(parts)
                - len(scored_parts)
            ),
        },
        "total_score": (
            grades.get(
                "total_score"
            )
            if score_available
            else None
        ),
        "total_max_score": (
            grades.get(
                "total_max",
                grades.get(
                    "total_max_score"
                ),
            )
            if score_available
            else None
        ),
        "parts": parts,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / output_name

    out_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out_path


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value or "0")

    if number.is_integer():
        return str(int(number))

    return (
        f"{number:.2f}"
        .rstrip("0")
        .rstrip(".")
    )


def _strip_document_wrapper(
    value: str,
) -> str:
    text = str(value or "").strip()

    begin_document = (
        r"\begin{document}"
    )

    end_document = (
        r"\end{document}"
    )

    if begin_document in text:
        text = text.split(
            begin_document,
            1,
        )[1]

    if end_document in text:
        text = text.rsplit(
            end_document,
            1,
        )[0]

    return text.strip()


def _safe_feedback_text(
    value: Any,
) -> str:
    """
    Render AI feedback as prose and math, while dropping presentation
    commands that must not leak into the structured result.
    """
    text = str(value or "").strip()

    if not text:
        return ""

    text = re.sub(
        r"\\(?:vspace|hspace)\*?"
        r"(?:\[[^\]]*\])?"
        r"\{[^{}]*\}",
        " ",
        text,
    )

    text = re.sub(
        r"\\(?:hrule|par|smallskip|"
        r"medskip|bigskip|newpage|"
        r"clearpage)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()

    normalized = normalize_math_text(
        text
    )

    return latex_render_mixed(
        normalized
    )


def _append_feedback_items(
    lines: list[str],
    title: str,
    items: list[str],
    color_name: str,
) -> None:
    if not items:
        return

    lines.extend(
        [
            (
                rf"\textcolor{{{color_name}}}"
                rf"{{\textbf{{{title}}}}}"
            ),
            r"\begin{itemize}",
        ]
    )

    for item in items:
        lines.append(
            rf"\item {_safe_feedback_text(item)}"
        )

    lines.append(
        r"\end{itemize}"
    )


def build_graded_result_tex(
    *,
    graded_result_json: Path,
    out_dir: Path,
    output_stem: str = "graded_result",
    font_name: str = "Arial",
) -> Path:
    """
    Render the canonical result as:

      Question
      Student answer
      Feedback

    repeated once for every question/part.
    """
    data = json.loads(
        Path(
            graded_result_json
        ).read_text(
            encoding="utf-8"
        )
    )

    parts = (
        data.get("parts")
        if isinstance(
            data.get("parts"),
            list,
        )
        else []
    )

    lines: list[str] = [
        r"\documentclass[12pt]{article}",
        (
            r"\usepackage"
            r"[a4paper,margin=1.8cm]"
            r"{geometry}"
        ),
        r"\usepackage{fontspec}",
        rf"\setmainfont{{{font_name}}}",
        r"\usepackage{polyglossia}",
        (
            r"\setdefaultlanguage"
            r"{hebrew}"
        ),
        (
            r"\setotherlanguage"
            r"{english}"
        ),
        (
            rf"\newfontfamily"
            rf"\hebrewfont"
            rf"{{{font_name}}}"
        ),
        (
            rf"\newfontfamily"
            rf"\englishfont"
            rf"{{{font_name}}}"
        ),
        (
            r"\usepackage"
            r"{amsmath,amssymb,mathtools}"
        ),
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        (
            r"\usepackage"
            r"[most]{tcolorbox}"
        ),
        (
            r"\definecolor"
            r"{MGOrange}{HTML}{F39A1E}"
        ),
        (
            r"\definecolor"
            r"{MGGreen}{HTML}{2F7D32}"
        ),
        (
            r"\definecolor"
            r"{MGRed}{HTML}{C43D2F}"
        ),
        (
            r"\definecolor"
            r"{MGText}{HTML}{4F4F52}"
        ),
        (
            r"\definecolor"
            r"{MGBorder}{HTML}{E6DED3}"
        ),
        (
            r"\setlength"
            r"{\parindent}{0pt}"
        ),
        (
            r"\setlength"
            r"{\parskip}{0.55em}"
        ),
        (
            r"\setlist[itemize]"
            r"{leftmargin=*,"
            r"itemsep=2pt,"
            r"topsep=3pt}"
        ),
        r"\raggedbottom",
        r"\begin{document}",
        r"\begin{center}",
        (
            r"{\LARGE\bfseries "
            r"משוב בדיקה}\par"
        ),
        r"\end{center}",
    ]

    exam_id = str(
        data.get("exam_id")
        or ""
    ).strip()

    provider = str(
        data.get("provider")
        or ""
    ).strip()

    total_score = data.get(
        "total_score"
    )

    total_max = data.get(
        "total_max_score"
    )

    summary_bits: list[str] = []

    if exam_id:
        summary_bits.append(
            (
                r"\textbf{מטלה:} "
                r"\begin{english}"
                f"{_tex_escape_title(exam_id)}"
                r"\end{english}"
            )
        )

    if provider:
        summary_bits.append(
            (
                r"\textbf{מודל בדיקה:} "
                r"\begin{english}"
                f"{_tex_escape_title(provider)}"
                r"\end{english}"
            )
        )

    if total_max not in (
        None,
        "",
        0,
        0.0,
    ):
        summary_bits.append(
            (
                r"\textbf{ציון כולל:} "
                f"{_format_number(total_score)}"
                " / "
                f"{_format_number(total_max)}"
            )
        )

    if summary_bits:
        lines.extend(
            [
                (
                    r"\begin{tcolorbox}"
                    r"[colback=orange!5,"
                    r"colframe=MGOrange!45,"
                    r"arc=3mm,"
                    r"boxrule=0.7pt]"
                ),
                r"\\[2pt]".join(
                    summary_bits
                ),
                r"\end{tcolorbox}",
            ]
        )

    if not parts:
        lines.append(
            (
                r"\textcolor{MGRed}"
                r"{לא נמצאו חלקים להצגה.}"
            )
        )

    for index, part in enumerate(
        parts,
        start=1,
    ):
        if not isinstance(part, dict):
            continue

        qid = str(
            part.get("qid")
            or f"Q{index}"
        ).strip()

        question_text = (
            _strip_document_wrapper(
                str(
                    part.get(
                        "question_text"
                    )
                    or ""
                )
            )
        )

        student_answer = (
            _strip_document_wrapper(
                str(
                    part.get(
                        "student_answer"
                    )
                    or ""
                )
            )
        )

        score = part.get(
            "score",
            0,
        )

        max_score = part.get(
            "max_score",
            0,
        )

        show_score = (
                part.get(
                    "scoring_mode"
                ) == "scored"
                and _positive_number(
            max_score
        )
                is not None
        )

        summary = str(
            part.get("summary")
            or ""
        ).strip()

        lines.extend(
            [
                (
                    r"\begin{tcolorbox}"
                    r"[enhanced,"
                    r"breakable,"
                    r"colback=white,"
                    r"colframe=MGBorder,"
                    r"arc=4mm,"
                    r"boxrule=0.8pt,"
                    r"left=5mm,"
                    r"right=5mm,"
                    r"top=4mm,"
                    r"bottom=4mm]"
                ),
                (
                    r"\begin{english}"
                    r"\textbf{\large "
                    f"{_tex_escape_title(qid)}"
                    r"}"
                    r"\end{english}"
                ),
                *(
                    [
                        r"\hfill",
                        (
                            r"\textcolor{MGGreen}"
                            r"{\textbf{"
                            f"{_format_number(score)}"
                            " / "
                            f"{_format_number(max_score)}"
                            r"}}\par"
                        ),
                    ]
                    if show_score
                    else [
                        r"\par",
                    ]
                ),
                (
                    r"\medskip"
                    r"\textcolor{MGOrange}"
                    r"{\textbf{שאלה}}\par"
                ),
                (
                    question_text
                    if question_text
                    else (
                        r"\textcolor{MGRed}"
                        r"{טקסט השאלה אינו זמין.}"
                    )
                ),
                (
                    r"\medskip"
                    r"\textcolor{MGOrange}"
                    r"{\textbf{תשובת הסטודנט}}"
                    r"\par"
                ),
                (
                    student_answer
                    if student_answer
                    else (
                        r"\textcolor{MGRed}"
                        r"{לא זוהתה תשובת סטודנט.}"
                    )
                ),
                (
                    r"\medskip"
                    r"\textcolor{MGGreen}"
                    r"{\textbf{משוב}}\par"
                ),
            ]
        )

        if summary:
            lines.append(
                _safe_feedback_text(
                    summary
                )
                + r"\par"
            )

        _append_feedback_items(
            lines,
            "מה נעשה נכון",
            _as_string_list(
                part.get(
                    "what_was_correct"
                )
            ),
            "MGGreen",
        )

        _append_feedback_items(
            lines,
            "טעויות או חלקים חסרים",
            _as_string_list(
                part.get(
                    "main_mistakes"
                )
            ),
            "MGRed",
        )

        _append_feedback_items(
            lines,
            "איך להשתפר",
            _as_string_list(
                part.get(
                    "how_to_improve"
                )
            ),
            "MGOrange",
        )

        next_step = str(
            part.get(
                "suggested_next_step"
            )
            or ""
        ).strip()

        if next_step:
            lines.extend(
                [
                    (
                        r"\begin{tcolorbox}"
                        r"[colback=orange!4,"
                        r"colframe=MGOrange!35,"
                        r"arc=2mm,"
                        r"boxrule=0.5pt,"
                        r"title={צעד מומלץ הבא}]"
                    ),
                    _safe_feedback_text(
                        next_step
                    ),
                    r"\end{tcolorbox}",
                ]
            )

        mismatch = (
            part.get("mismatch")
            if isinstance(
                part.get("mismatch"),
                dict,
            )
            else {}
        )

        if mismatch.get("is_mismatch"):
            explanation = str(
                mismatch.get(
                    "explanation_he"
                )
                or ""
            ).strip()

            lines.extend(
                [
                    (
                        r"\begin{tcolorbox}"
                        r"[colback=red!3,"
                        r"colframe=MGRed!45,"
                        r"arc=2mm,"
                        r"boxrule=0.5pt,"
                        r"title={אי התאמה אפשרית}]"
                    ),
                    _safe_feedback_text(
                        explanation
                        or (
                            "ייתכן שהתשובה "
                            "מתייחסת לשאלה אחרת."
                        )
                    ),
                    r"\end{tcolorbox}",
                ]
            )

        lines.extend(
            [
                r"\end{tcolorbox}",
                r"\medskip",
            ]
        )

    lines.extend(
        [
            r"\end{document}",
            "",
        ]
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / f"{output_stem}.tex"
    )

    out_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return out_path