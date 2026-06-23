from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.ai_clients.gpt_client import GptClient
from common.tex.part_normalize import normalize_part

from schemas.reference_bundle import (
    AnswersOnlyBundle,
    FullSolutionBundle,
    QuestionsAnswersBundle,
    QuestionsOnlyBundle,
    answers_only_bundle_schema,
    full_solution_bundle_schema,
    questions_answers_bundle_schema,
    questions_only_bundle_schema,
)


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


QUESTIONS_ONLY_SYSTEM_PROMPT = """
You are MathGrade Questions-Only Builder.

Input:
- OCR text from a teacher-uploaded mathematics exam file.
- The file should contain questions, but not necessarily answers.

Output:
- Return strict JSON matching the QuestionsOnlyBundle schema.
- The output should contain only the exam structure and question text:
  question_id, parts, question_text, required_action, optional max_points.
- Do not fill official_solution, expected_answer, or grading instructions here.

Rules:
- Extract only questions intended for submission.
- Ignore headers, page numbers, dates, instructions not relevant to grading, and sections marked as not for submission.
- Preserve Hebrew text.
- Preserve mathematics as LaTeX where possible.
- Detect Hebrew part labels: א, ב, ג, ד, ה, ו, ז, ח, ט, י.
- part_key must be normalized Latin: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
- If a question has no explicit parts, represent it as one part: part="א", part_key="a".
- If a shared stem applies to several parts, repeat that stem inside each part's question_text.
- required_action should describe what the student is expected to do.
- If uncertain, keep a conservative reconstruction and add a warning.
"""


ANSWERS_ONLY_SYSTEM_PROMPT = """
You are MathGrade Answers-Only Builder.

Input:
- OCR text from a teacher-uploaded official answers / solutions file.
- The file may contain only answers, not the full question text.

Output:
- Return strict JSON matching the AnswersOnlyBundle schema.
- The output should contain extracted official_solution, expected_answer, and grading_instructions.
- Include question_id and part/part_key when visible or strongly inferable.
- If the file contains short hints of the question text, place them in question_hint.

Rules:
- Do not invent missing question text.
- Do not invent official solutions that are not supported by the source.
- If a question number or part label is unclear, set review_status="uncertain" and add a warning.
- expected_answer should be concise: final result, interval, proof target, counterexample, theorem, etc.
- grading_instructions should help the grading AI grade fairly.
- Accept mathematically equivalent methods; do not require exact official wording.
- Preserve Hebrew and LaTeX where appropriate.
- part_key must be normalized Latin: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
"""


QUESTIONS_ANSWERS_SYSTEM_PROMPT = """
You are MathGrade Questions-And-Answers Builder.

Input:
- OCR text from a teacher-uploaded mathematics file that contains both questions and official answers/solutions.

Output:
- Return strict JSON matching the QuestionsAnswersBundle schema.
- The output should contain question_text, required_action, official_solution, expected_answer, and grading_instructions.

Rules:
- Extract only questions intended for submission.
- Ignore page headers, dates, page numbers, repeated titles, and administrative text.
- Stop before sections marked as not for submission.
- Preserve Hebrew text.
- Preserve mathematics as LaTeX where possible.
- Detect Hebrew part labels: א, ב, ג, ד, ה, ו, ז, ח, ט, י.
- part_key must be normalized Latin: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
- If a question has no explicit parts, represent it as one part: part="א", part_key="a".
- If a shared stem applies to several parts, repeat that stem inside each part's question_text.
- Match each official solution to the correct question part.
- If answer alignment is unclear, keep the best conservative reconstruction, set review_status="needs_review", and add a warning.
- expected_answer should be concise.
- grading_instructions should help the grading AI grade student work fairly.
"""


FULL_SOLUTION_SYSTEM_PROMPT = """
You are MathGrade Full-Solution Builder.

Input:
- Structured questions and structured answers, or a combined questions+answers structure.

Output:
- Return strict JSON matching the FullSolutionBundle schema.
- The output is the final gradeable solution-bank object.

Rules:
- Every output part should contain:
  question_text, required_action, official_solution, expected_answer, grading_instructions.
- Preserve question_id, part, and part_key.
- Do not invent unsupported answers.
- If an answer is missing, leave official_solution empty and mark review_status="missing".
- If question-answer alignment is uncertain, mark review_status="needs_review" or "conflict".
- Preserve Hebrew and LaTeX.
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


def _safe_model_dump(model_or_dict: Any) -> dict[str, Any]:
    if hasattr(model_or_dict, "model_dump"):
        return model_or_dict.model_dump()
    if isinstance(model_or_dict, dict):
        return model_or_dict
    return {}


def _normalize_question_id(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _part_key(part: str, part_key: str = "") -> str:
    return str(part_key or "").strip() or normalize_part(part)


def _as_questions_only_bundle(data: dict[str, Any], *, exam_id: str = "") -> QuestionsOnlyBundle:
    if exam_id and not data.get("exam_id"):
        data["exam_id"] = exam_id
    return QuestionsOnlyBundle.model_validate(data)


def _as_answers_only_bundle(data: dict[str, Any], *, exam_id: str = "") -> AnswersOnlyBundle:
    if exam_id and not data.get("exam_id"):
        data["exam_id"] = exam_id
    return AnswersOnlyBundle.model_validate(data)


def _as_questions_answers_bundle(data: dict[str, Any], *, exam_id: str = "") -> QuestionsAnswersBundle:
    if exam_id and not data.get("exam_id"):
        data["exam_id"] = exam_id
    return QuestionsAnswersBundle.model_validate(data)


def _as_full_solution_bundle(data: dict[str, Any], *, exam_id: str = "") -> FullSolutionBundle:
    if exam_id and not data.get("exam_id"):
        data["exam_id"] = exam_id
    return FullSolutionBundle.model_validate(data)


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


def build_questions_only_bundle_from_ocr(
    *,
    ocr_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    """
    Input:
      OCR text from a questions-only teacher upload.

    Output:
      QuestionsOnlyBundle as dict.
    """
    client = client or GptClient()

    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

OCR source:
-----------
{ocr_text}
"""

    result = client.chat_json(
        system=QUESTIONS_ONLY_SYSTEM_PROMPT,
        user=user_prompt,
        schema=questions_only_bundle_schema(),
        schema_name="mathgrade_questions_only_bundle",
        strict=False,
        timeout_s=240,
    )

    return _as_questions_only_bundle(result, exam_id=exam_id).model_dump()


def build_answers_only_bundle_from_ocr(
    *,
    ocr_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    """
    Input:
      OCR text from an answers-only / official-solutions teacher upload.

    Output:
      AnswersOnlyBundle as dict.
    """
    client = client or GptClient()

    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

OCR source:
-----------
{ocr_text}
"""

    result = client.chat_json(
        system=ANSWERS_ONLY_SYSTEM_PROMPT,
        user=user_prompt,
        schema=answers_only_bundle_schema(),
        schema_name="mathgrade_answers_only_bundle",
        strict=False,
        timeout_s=240,
    )

    return _as_answers_only_bundle(result, exam_id=exam_id).model_dump()


def build_questions_answers_bundle_from_ocr(
    *,
    ocr_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    """
    Input:
      OCR text from a combined questions+answers teacher upload.

    Output:
      QuestionsAnswersBundle as dict.
    """
    client = client or GptClient()

    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

OCR source:
-----------
{ocr_text}
"""

    result = client.chat_json(
        system=QUESTIONS_ANSWERS_SYSTEM_PROMPT,
        user=user_prompt,
        schema=questions_answers_bundle_schema(),
        schema_name="mathgrade_questions_answers_bundle",
        strict=False,
        timeout_s=240,
    )

    return _as_questions_answers_bundle(result, exam_id=exam_id).model_dump()


def promote_questions_answers_to_full_solution(
    *,
    questions_answers_bundle: dict[str, Any] | QuestionsAnswersBundle,
    exam_id: str = "",
) -> dict[str, Any]:
    """
    Input:
      QuestionsAnswersBundle from a combined questions+answers upload.

    Output:
      FullSolutionBundle as dict.
    """
    qab = (
        questions_answers_bundle
        if isinstance(questions_answers_bundle, QuestionsAnswersBundle)
        else QuestionsAnswersBundle.model_validate(questions_answers_bundle)
    )

    questions: list[dict[str, Any]] = []

    for q in qab.questions:
        parts: list[dict[str, Any]] = []

        for p in q.parts:
            part_key = _part_key(p.part, p.part_key)
            warnings = list(p.warnings or [])

            review_status = p.review_status
            if not p.official_solution and not p.expected_answer:
                review_status = "missing"
                warnings.append("No official solution or expected answer was extracted for this part.")

            parts.append(
                {
                    "part": p.part,
                    "part_key": part_key,
                    "question_text": p.question_text,
                    "required_action": p.required_action,
                    "official_solution": p.official_solution,
                    "expected_answer": p.expected_answer,
                    "grading_instructions": p.grading_instructions,
                    "max_points": p.max_points,
                    "review_status": review_status,
                    "confidence": p.confidence,
                    "warnings": warnings,
                    "source_question_file": ",".join(qab.source_names or []),
                    "source_answer_file": ",".join(qab.source_names or []),
                }
            )

        questions.append(
            {
                "question_id": q.question_id,
                "parts": parts,
            }
        )

    full = FullSolutionBundle(
        exam_id=qab.exam_id or exam_id,
        exam_title=qab.exam_title,
        questions=questions,
        warnings=list(qab.warnings or []),
        structure_corrections=list(qab.structure_corrections or []),
        source_names=list(qab.source_names or []),
    )

    return full.model_dump()


def merge_questions_and_answers_to_full_solution(
    *,
    questions_bundle: dict[str, Any] | QuestionsOnlyBundle,
    answers_bundle: dict[str, Any] | AnswersOnlyBundle,
    exam_id: str = "",
) -> dict[str, Any]:
    """
    Input:
      QuestionsOnlyBundle + AnswersOnlyBundle.

    Output:
      FullSolutionBundle as dict.

    Matching:
      Primary key: question_id + part_key.
      If an answer part is missing, the output part is marked missing.
      If an answer exists without a matching question, a warning is added.
    """
    qb = (
        questions_bundle
        if isinstance(questions_bundle, QuestionsOnlyBundle)
        else QuestionsOnlyBundle.model_validate(questions_bundle)
    )
    ab = (
        answers_bundle
        if isinstance(answers_bundle, AnswersOnlyBundle)
        else AnswersOnlyBundle.model_validate(answers_bundle)
    )

    answer_index: dict[tuple[int, str], Any] = {}
    unmatched_answer_keys: set[tuple[int, str]] = set()

    for aq in ab.questions:
        qid = _normalize_question_id(aq.question_id)
        if qid is None:
            continue

        for ap in aq.parts:
            pk = _part_key(ap.part, ap.part_key)
            if not pk:
                continue

            key = (qid, pk)
            answer_index[key] = ap
            unmatched_answer_keys.add(key)

    full_questions: list[dict[str, Any]] = []
    warnings: list[str] = []
    warnings.extend(qb.warnings or [])
    warnings.extend(ab.warnings or [])

    for q in qb.questions:
        full_parts: list[dict[str, Any]] = []

        for p in q.parts:
            pk = _part_key(p.part, p.part_key)
            key = (q.question_id, pk)

            ap = answer_index.get(key)
            part_warnings = list(p.warnings or [])

            if ap:
                unmatched_answer_keys.discard(key)
                part_warnings.extend(ap.warnings or [])

                official_solution = ap.official_solution
                expected_answer = ap.expected_answer
                grading_instructions = ap.grading_instructions
                review_status = ap.review_status
                confidence = ap.confidence
                answer_source = ",".join(ab.source_names or [])
            else:
                official_solution = ""
                expected_answer = ""
                grading_instructions = ""
                review_status = "missing"
                confidence = p.confidence
                answer_source = ""
                part_warnings.append("No matching answer was found for this question part.")

            full_parts.append(
                {
                    "part": p.part,
                    "part_key": pk,
                    "question_text": p.question_text,
                    "required_action": p.required_action,
                    "official_solution": official_solution,
                    "expected_answer": expected_answer,
                    "grading_instructions": grading_instructions,
                    "max_points": p.max_points,
                    "review_status": review_status,
                    "confidence": confidence,
                    "warnings": part_warnings,
                    "source_question_file": ",".join(qb.source_names or []),
                    "source_answer_file": answer_source,
                }
            )

        full_questions.append(
            {
                "question_id": q.question_id,
                "parts": full_parts,
            }
        )

    for qid, pk in sorted(unmatched_answer_keys):
        warnings.append(f"Answer exists for Q{qid}{pk}, but no matching question part was found.")

    full = FullSolutionBundle(
        exam_id=qb.exam_id or ab.exam_id or exam_id,
        exam_title=qb.exam_title or ab.exam_title,
        questions=full_questions,
        warnings=warnings,
        structure_corrections=[
            *(qb.structure_corrections or []),
            *(ab.structure_corrections or []),
        ],
        source_names=[
            *(qb.source_names or []),
            *(ab.source_names or []),
        ],
    )

    return full.model_dump()


def build_questions_bundle_from_mathpix(
    *,
    mathpix_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper.

    Input:
      OCR text from a questions-only file.

    Output:
      QuestionsOnlyBundle-compatible dict.
    """
    return build_questions_only_bundle_from_ocr(
        ocr_text=mathpix_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
    )


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