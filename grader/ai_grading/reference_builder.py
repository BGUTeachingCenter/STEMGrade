from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grader.ai_clients.gpt_client import GptClient
from grader.file_handling.part_normalize import normalize_part


REFERENCE_BUNDLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["exam_title", "questions", "warnings"],
    "properties": {
        "exam_title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question_id", "stem", "parts"],
                "properties": {
                    "question_id": {"type": "integer"},
                    "stem": {"type": "string"},
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "part",
                                "part_key",
                                "question_text",
                                "required_action",
                                "expected_answer",
                                "grading_instructions",
                            ],
                            "properties": {
                                "part": {"type": "string"},
                                "part_key": {"type": "string"},
                                "question_text": {"type": "string"},
                                "required_action": {"type": "string"},
                                "expected_answer": {"type": "string"},
                                "grading_instructions": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


SYSTEM_PROMPT = """
You are MathGrade Reference Builder.

Your job is to convert noisy Mathpix OCR / Mathpix Markdown from a Hebrew mathematics exam
into a clean structured JSON bundle for automated grading.

Rules:
- Return JSON only.
- Extract only questions that are intended for submission.
- Ignore page headers, university headers, dates, page numbers, and repeated titles.
- Stop before sections like "שאלות שאינן להגשה" or optional starred questions not for submission.
- Preserve Hebrew question text.
- Preserve mathematics as LaTeX where possible.
- Detect Hebrew part labels: א, ב, ג, ד, ה, ו, ז, ח, ט, י.
- For each part, include the shared question stem if needed for understanding.
- Do not invent a question or part that is not supported by the source.
- If OCR is ambiguous, keep the best reconstruction and add a warning.
- The part_key must be latin normalized: א=a, ב=b, ג=c, ד=d, ה=e, ו=f.
- expected_answer and grading_instructions may be empty for questions-only uploads, but if the expected answer is obvious from the prompt, you may include a short note.
"""


def _normalize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Defensive cleanup after model output.
    """
    questions = bundle.get("questions") or []
    clean_questions: list[dict[str, Any]] = []

    for q in questions:
        try:
            qid = int(q.get("question_id"))
        except Exception:
            continue

        stem = str(q.get("stem") or "").strip()
        parts = q.get("parts") or []

        clean_parts: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for p in parts:
            part = str(p.get("part") or "").strip()
            part_key = str(p.get("part_key") or "").strip() or normalize_part(part)

            if not part_key:
                continue

            if part_key in seen_keys:
                continue
            seen_keys.add(part_key)

            clean_parts.append(
                {
                    "part": part,
                    "part_key": part_key,
                    "question_text": str(p.get("question_text") or "").strip(),
                    "required_action": str(p.get("required_action") or "").strip(),
                    "expected_answer": str(p.get("expected_answer") or "").strip(),
                    "grading_instructions": str(p.get("grading_instructions") or "").strip(),
                }
            )

        clean_questions.append(
            {
                "question_id": qid,
                "stem": stem,
                "parts": clean_parts,
            }
        )

    clean_questions.sort(key=lambda x: int(x["question_id"]))

    return {
        "exam_title": str(bundle.get("exam_title") or "").strip(),
        "questions": clean_questions,
        "warnings": [str(x) for x in (bundle.get("warnings") or [])],
    }


def build_questions_bundle_from_mathpix(
    *,
    mathpix_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    client = client or GptClient()

    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

Mathpix OCR / MMD source:
-------------------------
{mathpix_text}
"""

    result = client.chat_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        schema=REFERENCE_BUNDLE_SCHEMA,
        schema_name="mathgrade_reference_bundle",
        strict=True,
        timeout_s=180,
    )

    return _normalize_bundle(result)


def write_questions_bundle_json(bundle: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def questions_bundle_to_tex(bundle: dict[str, Any]) -> str:
    """
    Convert validated JSON bundle into canonical TeX that parse_reference_tex() can read.

    Important:
    parse_reference_tex expects:
      \\section*{Question N}
      \\subsection*{(א)}
    """
    lines: list[str] = [
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
        r"% Canonical MathGrade questions-only file generated by AI Reference Builder.",
        "",
    ]

    title = str(bundle.get("exam_title") or "").strip()
    if title:
        lines.append(rf"\textbf{{{title}}}")
        lines.append("")

    for q in bundle.get("questions") or []:
        qid = q.get("question_id")
        stem = str(q.get("stem") or "").strip()

        lines.append("")
        lines.append(rf"\section*{{Question {qid}}}")

        parts = q.get("parts") or []
        if not parts:
            lines.append("")
            lines.append(r"\subsection*{(a)}")
            if stem:
                lines.append(stem)
                lines.append("")
            lines.append("% No parts detected.")
            continue

        for p in parts:
            part = str(p.get("part") or "").strip()
            part_key = str(p.get("part_key") or "").strip()
            question_text = str(p.get("question_text") or "").strip()
            required_action = str(p.get("required_action") or "").strip()
            expected_answer = str(p.get("expected_answer") or "").strip()
            grading_instructions = str(p.get("grading_instructions") or "").strip()

            display_part = part or part_key or "a"

            lines.append("")
            lines.append(rf"\subsection*{{({display_part})}}")

            if stem:
                lines.append(stem)
                lines.append("")

            if question_text:
                lines.append(question_text)
                lines.append("")

            if required_action:
                lines.append(r"\paragraph{Required action}")
                lines.append(required_action)
                lines.append("")

            if expected_answer:
                lines.append(r"\paragraph{Expected answer}")
                lines.append(expected_answer)
                lines.append("")

            if grading_instructions:
                lines.append(r"\paragraph{Grading instructions}")
                lines.append(grading_instructions)
                lines.append("")

    lines.extend(["", r"\end{document}", ""])
    return "\n".join(lines)