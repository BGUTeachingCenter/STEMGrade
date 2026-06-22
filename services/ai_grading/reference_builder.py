from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.ai_clients.gpt_client import GptClient
from services.file_handling.part_normalize import normalize_part


REFERENCE_BUNDLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["exam_id", "exam_title", "questions", "warnings", "structure_corrections"],
    "properties": {
        "exam_id": {"type": "string"},
        "exam_title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question_id", "parts"],
                "properties": {
                    "question_id": {"type": "integer"},
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
                                "official_solution",
                                "expected_answer",
                                "grading_instructions",
                            ],
                            "properties": {
                                "part": {"type": "string"},
                                "part_key": {"type": "string"},
                                "question_text": {"type": "string"},
                                "required_action": {"type": "string"},
                                "official_solution": {"type": "string"},
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
        "structure_corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "description", "confidence"],
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string"},
                },
            },
        },
    },
}


QUESTIONS_SYSTEM_PROMPT = """
You are MathGrade Exam Builder.

Your job is to convert noisy Mathpix OCR / Mathpix Markdown from a Hebrew mathematics exam
into a clean structured JSON bundle for automated grading.

Return strict JSON only.

General rules:
- Extract only questions intended for submission.
- Ignore page headers, university names, dates, page numbers, repeated titles, and administrative text.
- Stop before sections like "שאלות שאינן להגשה" or optional/starred questions not for submission.
- Preserve Hebrew question text.
- Preserve mathematics as LaTeX where possible.
- Detect Hebrew part labels: א, ב, ג, ד, ה, ו, ז, ח, ט, י.
- Correct obvious Mathpix OCR mistakes in part labels only when context is clear.
- Do not overfit to one exam. Exams can have different numbering, wording, and part structures.
- Preserve visible question numbering. Do not merge a numbered question into the previous question.
- If a question has no explicit parts, represent it as one part: part="א", part_key="a".
- For each part, include the complete question text needed for grading.
- If a shared stem applies to several parts, repeat that stem inside each part's question_text.
- Do not invent questions or parts not supported by the source.
- required_action should describe what the student is expected to do.
- For questions-only uploads, leave official_solution, expected_answer, and grading_instructions empty unless extremely obvious.
- part_key must be normalized Latin: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
- If uncertain, keep the best conservative reconstruction and add a warning.
- structure_corrections should be empty for the initial questions-only build unless you corrected an obvious OCR structural mistake.
"""


REFERENCE_SYSTEM_PROMPT = """
You are MathGrade Solution Aligner.

Your job is to align noisy Mathpix OCR / Mathpix Markdown from an official solution file
to an existing clean question bundle.

Return strict JSON only.

Rules:
- Use the existing question bundle as the primary structure.
- Keep question_id, part, part_key, question_text, and required_action from the existing bundle unless the official solution clearly proves a structure error.
- Fill official_solution, expected_answer, and grading_instructions for each part.
- Use the official solution source as evidence. Do not invent a solution if it is not present.
- If a solution is missing or unclear, leave official_solution empty and add a warning.
- expected_answer should be concise: final result, theorem, proof target, interval, counterexample, etc.
- grading_instructions should help the grading AI grade student work fairly.
- Accept mathematically equivalent methods; do not require the exact official wording.
- Preserve mathematics as LaTeX.
- Preserve Hebrew where appropriate.
- You may correct question boundaries and part labels only when the official solution clearly supports the correction.
- Record every structural correction in structure_corrections.
- If you correct structure, output the corrected full bundle.
- Do not add optional/not-for-submission questions unless the existing bundle already contains them.
"""


def _empty_part() -> dict[str, str]:
    return {
        "part": "",
        "part_key": "",
        "question_text": "",
        "required_action": "",
        "official_solution": "",
        "expected_answer": "",
        "grading_instructions": "",
    }


def _normalize_part_item(p: dict[str, Any]) -> dict[str, str]:
    item = _empty_part()

    item["part"] = str(p.get("part") or "").strip()
    item["part_key"] = str(p.get("part_key") or "").strip() or normalize_part(item["part"])
    item["question_text"] = str(p.get("question_text") or "").strip()
    item["required_action"] = str(p.get("required_action") or "").strip()
    item["official_solution"] = str(p.get("official_solution") or "").strip()
    item["expected_answer"] = str(p.get("expected_answer") or "").strip()
    item["grading_instructions"] = str(p.get("grading_instructions") or "").strip()

    if not item["part"] and item["part_key"]:
        item["part"] = item["part_key"]

    return item


def normalize_reference_bundle(bundle: dict[str, Any], *, exam_id: str = "") -> dict[str, Any]:
    questions = bundle.get("questions") or []
    clean_questions: list[dict[str, Any]] = []

    for q in questions:
        try:
            qid = int(q.get("question_id"))
        except Exception:
            continue

        clean_parts: list[dict[str, str]] = []
        seen_keys: set[str] = set()

        for p in q.get("parts") or []:
            if not isinstance(p, dict):
                continue

            item = _normalize_part_item(p)
            key = item["part_key"]

            if not key or key in seen_keys:
                continue

            seen_keys.add(key)
            clean_parts.append(item)

        clean_questions.append(
            {
                "question_id": qid,
                "parts": clean_parts,
            }
        )

    clean_questions.sort(key=lambda x: int(x["question_id"]))

    corrections = []
    for c in bundle.get("structure_corrections") or []:
        if isinstance(c, dict):
            corrections.append(
                {
                    "type": str(c.get("type") or "").strip(),
                    "description": str(c.get("description") or "").strip(),
                    "confidence": str(c.get("confidence") or "").strip(),
                }
            )

    return {
        "exam_id": str(bundle.get("exam_id") or exam_id or "").strip(),
        "exam_title": str(bundle.get("exam_title") or "").strip(),
        "questions": clean_questions,
        "warnings": [str(x) for x in (bundle.get("warnings") or [])],
        "structure_corrections": corrections,
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
        system=QUESTIONS_SYSTEM_PROMPT,
        user=user_prompt,
        schema=REFERENCE_BUNDLE_SCHEMA,
        schema_name="mathgrade_questions_bundle",
        strict=True,
        timeout_s=240,
    )

    return normalize_reference_bundle(result, exam_id=exam_id)


def build_reference_bundle_from_mathpix(
    *,
    questions_bundle: dict[str, Any],
    solution_mathpix_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    client = client or GptClient()

    normalized_questions = normalize_reference_bundle(questions_bundle, exam_id=exam_id)

    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

Existing clean question bundle:
-------------------------------
{json.dumps(normalized_questions, ensure_ascii=False, indent=2)}

Official solution Mathpix OCR / MMD source:
------------------------------------------
{solution_mathpix_text}
"""

    result = client.chat_json(
        system=REFERENCE_SYSTEM_PROMPT,
        user=user_prompt,
        schema=REFERENCE_BUNDLE_SCHEMA,
        schema_name="mathgrade_reference_bundle",
        strict=True,
        timeout_s=240,
    )

    return normalize_reference_bundle(result, exam_id=exam_id)


def write_reference_bundle_json(bundle: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_questions_bundle_json(bundle: dict[str, Any], out_path: Path) -> None:
    write_reference_bundle_json(bundle, out_path)


def bundle_to_exam_structure(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_reference_bundle(bundle)

    questions = []
    for q in normalized.get("questions") or []:
        questions.append(
            {
                "question_id": str(q["question_id"]),
                "parts": [
                    p["part"]
                    for p in q.get("parts") or []
                    if p.get("part")
                ],
            }
        )

    return {
        "questions": questions,
        "question_count": len(questions),
        "part_count": sum(len(q.get("parts") or []) for q in questions),
    }


def questions_bundle_to_tex(bundle: dict[str, Any]) -> str:
    """
    Convert validated JSON bundle into canonical TeX that parse_reference_tex() can read.

    parse_reference_tex expects:
      \\section*{Question N}
      \\subsection*{(א)}
    """
    normalized = normalize_reference_bundle(bundle)

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
        r"% Canonical MathGrade file generated from structured JSON.",
        "",
    ]

    title = str(normalized.get("exam_title") or "").strip()
    if title:
        lines.append(rf"\textbf{{{title}}}")
        lines.append("")

    for q in normalized.get("questions") or []:
        qid = q.get("question_id")
        lines.append("")
        lines.append(rf"\section*{{Question {qid}}}")

        for p in q.get("parts") or []:
            display_part = p.get("part") or p.get("part_key") or "a"

            question_text = p.get("question_text", "").strip()
            required_action = p.get("required_action", "").strip()
            official_solution = p.get("official_solution", "").strip()
            expected_answer = p.get("expected_answer", "").strip()
            grading_instructions = p.get("grading_instructions", "").strip()

            lines.append("")
            lines.append(rf"\subsection*{{({display_part})}}")

            if question_text:
                lines.append(question_text)
                lines.append("")

            if required_action:
                lines.append(r"\paragraph{Required action}")
                lines.append(required_action)
                lines.append("")

            if official_solution:
                lines.append(r"\paragraph{Official solution}")
                lines.append(official_solution)
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