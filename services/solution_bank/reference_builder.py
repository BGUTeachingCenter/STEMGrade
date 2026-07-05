from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from core.ai_clients.gpt_client import GptClient
from common.tex.part_normalize import normalize_part
from common.tex.question_segmenter import segment_document

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
- Extract all visible questions, including regular submitted questions and sections marked as not for submission / optional / practice.
- Ignore headers, page numbers, dates, and administrative instructions that are not question content.
- If a question appears after a marker such as "שאלות שאינן להגשה", "לא להגשה", "שאלות רשות", "optional", or "not for submission":
  set is_gradeable=false, usage="practice_feedback_only", and section_label to the visible section marker.
- For normal submitted questions:
  set is_gradeable=true and usage="gradeable".
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
- The visible numbering in the answer file may be local numbering such as 1), 2), 3), and may NOT match the exam's global question_id.

Output:
- Return strict JSON matching the AnswersOnlyBundle schema.
- Extract answer candidates conservatively.
- Include official_solution, expected_answer, grading_instructions.
- Use question_hint to store any visible math expression or short text that identifies what this answer solves.

Critical rule:
- Do NOT force question_id or part_key unless the answer file explicitly shows a global question number and part label.
- If the file only shows local numbering like 1), 2), 3), keep question_id null or uncertain rather than pretending it is Question 1, Question 2, Question 3.
- Prefer preserving source order and question_hint over inventing structure.

Rules:
- Do not invent missing question text.
- Do not invent official solutions that are not supported by the source.
- If a question number or part label is unclear, set review_status="uncertain" and add a warning.
- expected_answer should be concise: final result, interval, proof target, counterexample, theorem, etc.
- grading_instructions should help the grading AI grade fairly.
- Accept mathematically equivalent methods; do not require exact official wording.
- Preserve Hebrew and LaTeX where appropriate.
- If a solution begins with a visible expression such as |x-2|<=5, place that expression in question_hint.
- If a solution begins with a local number such as 1), 2), 3), mention it in warnings or question_hint, but do not treat it as global question_id unless the surrounding text clearly says it is a global question.
- part_key must be normalized Latin only when a real part label is visible: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
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


# ---------------------------------------------------------------------
# Answer-to-question alignment helpers
# ---------------------------------------------------------------------
# Why this exists:
# Answers-only files often use local labels like "1)", "2)", "3)".
# Those labels do not always mean Question 1, Question 2, Question 3.
# So we must align by mathematical/content overlap, not only by qid+part_key.


_HEBREW_STOPWORDS = {
    "את", "של", "על", "עם", "כל", "לכל", "כי", "אם", "אז", "או", "וגם",
    "הבא", "הבאה", "הוכיחו", "הוכח", "מצאו", "מצא", "סרטטו", "סרטט",
    "תשובתכם", "הטענה", "באמצעות", "דוגמה", "נגדית", "מתקיים", "מתקימת",
    "קבוצת", "המספרים", "המקיימים", "אי", "שוויון", "האי", "הפתרונות",
}


NON_SUBMISSION_MARKERS = [
    "שאלות שאינן להגשה",
    "שאלות שאינן להגשה:",
    "שאלות רשות",
    "שאלות בונוס",
    "לא להגשה",
    "אינו להגשה",
    "אינה להגשה",
    "אופציונלי",
    "optional questions",
    "optional exercises",
    "not for submission",
    "not to submit",
]


def split_submission_sections(ocr_text: str) -> tuple[str, str, str, list[str]]:
    """
    Split OCR text into:
      - gradeable/submitted section text
      - non-submission/practice section text
      - marker that caused the split
      - warnings

    We do NOT discard the non-submission section anymore. It is kept and later
    stored in the same JSON with is_gradeable=false.
    """
    text = str(ocr_text or "")
    if not text.strip():
        return text, "", "", []

    lowered = text.lower()
    best_idx: int | None = None
    best_marker = ""

    for marker in NON_SUBMISSION_MARKERS:
        idx = lowered.find(marker.lower())
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_marker = marker

    if best_idx is None:
        return text, "", "", []

    gradeable_text = text[:best_idx].rstrip()
    practice_text = text[best_idx:].strip()

    warnings = [
        f"Detected non-submission/practice section marker: {best_marker!r}.",
        "Questions after this marker were kept in the JSON as practice_feedback_only and excluded from reference_current.tex grading.",
    ]

    return gradeable_text, practice_text, best_marker, warnings


def trim_non_submission_sections(ocr_text: str) -> tuple[str, list[str]]:
    """
    Backward-compatible wrapper.

    Prefer split_submission_sections() for new code. This wrapper still returns
    only the gradeable text because older callers may depend on that behavior,
    but its warning must not claim that practice questions were deleted.
    """
    gradeable_text, _practice_text, marker, warnings = split_submission_sections(ocr_text)
    if marker:
        return gradeable_text, warnings
    return str(ocr_text or ""), []


def _question_ids_after_non_submission_marker(ocr_text: str) -> tuple[set[int], str, list[str]]:
    """
    Detect visible question numbers after a non-submission marker.

    Conservative generic patterns:
      *.7
      .7
      Question 7

    We avoid plain standalone numbers so random numbers inside the optional
    section do not become question IDs.
    """
    _gradeable_text, practice_text, marker, warnings = split_submission_sections(ocr_text)
    ids: set[int] = set()

    if not practice_text:
        return ids, marker, warnings

    patterns = [
        r"(?:^|\n)\s*\*+\s*\.?\s*(\d{1,3})(?=\s|[.)])",
        r"(?:^|\n)\s*\.(\d{1,3})(?=\s|[.)])",
        r"(?:^|\n)\s*question\s+(\d{1,3})(?=\s|[.)])",
        r"(?:^|\n)\s*שאלה\s+(\d{1,3})(?=\s|[.)])",
    ]

    for pat in patterns:
        for m in re.finditer(pat, practice_text, flags=re.IGNORECASE):
            try:
                qid = int(m.group(1))
            except Exception:
                continue
            if 1 <= qid <= 300:
                ids.add(qid)

    if marker and ids:
        warnings.append(
            "Detected practice/non-submission question IDs after marker "
            f"{marker!r}: {', '.join(str(x) for x in sorted(ids))}."
        )
    elif marker:
        warnings.append(
            f"Detected marker {marker!r}, but could not confidently detect practice question IDs after it."
        )

    return ids, marker, warnings


def apply_gradeable_flags_from_source(
    bundle: QuestionsOnlyBundle | QuestionsAnswersBundle | dict[str, Any],
    *,
    ocr_text: str,
) -> QuestionsOnlyBundle | QuestionsAnswersBundle:
    """
    Mark questions after a non-submission marker as practice_feedback_only.

    The same JSON can therefore contain:
      - gradeable questions
      - non-submission questions for later feedback-only use
    """
    if isinstance(bundle, (QuestionsOnlyBundle, QuestionsAnswersBundle)):
        out = bundle
    else:
        # Prefer QuestionsOnlyBundle because this helper is mainly used there.
        # If that fails, try QuestionsAnswersBundle.
        try:
            out = QuestionsOnlyBundle.model_validate(bundle)
        except Exception:
            out = QuestionsAnswersBundle.model_validate(bundle)

    practice_ids, marker, warnings = _question_ids_after_non_submission_marker(ocr_text)

    for q in out.questions:
        qid = int(q.question_id)
        question_is_practice = qid in practice_ids

        # Respect an AI-produced false flag too.
        if getattr(q, "is_gradeable", True) is False:
            question_is_practice = True

        if question_is_practice:
            q.is_gradeable = False
            q.usage = "practice_feedback_only"
            q.section_label = q.section_label or marker or "non-submission"
        else:
            q.is_gradeable = True
            q.usage = "gradeable"
            q.section_label = q.section_label or ""

        for p in q.parts:
            if question_is_practice or getattr(p, "is_gradeable", True) is False:
                p.is_gradeable = False
                p.usage = "practice_feedback_only"
                p.section_label = p.section_label or q.section_label or marker or "non-submission"
            else:
                p.is_gradeable = True
                p.usage = "gradeable"
                p.section_label = p.section_label or ""

    if warnings:
        out.warnings = list(dict.fromkeys([*(out.warnings or []), *warnings]))

    return out


def mark_bundle_as_practice_only(
    bundle: QuestionsOnlyBundle | dict[str, Any],
    *,
    section_label: str = "non-submission",
) -> QuestionsOnlyBundle:
    """
    Mark every question and part in a QuestionsOnlyBundle as practice-only.

    Used for questions extracted from a section such as:
      שאלות שאינן להגשה
    """
    qb = (
        bundle
        if isinstance(bundle, QuestionsOnlyBundle)
        else QuestionsOnlyBundle.model_validate(bundle)
    )

    for q in qb.questions:
        q.is_gradeable = False
        q.usage = "practice_feedback_only"
        q.section_label = q.section_label or section_label

        for p in q.parts:
            p.is_gradeable = False
            p.usage = "practice_feedback_only"
            p.section_label = p.section_label or q.section_label or section_label

    qb.warnings = list(
        dict.fromkeys(
            [
                *(qb.warnings or []),
                f"Questions in this section were marked as practice_feedback_only: {section_label}.",
            ]
        )
    )

    return qb


def combine_questions_only_bundles(
    *,
    gradeable_bundle: QuestionsOnlyBundle | dict[str, Any],
    practice_bundle: QuestionsOnlyBundle | dict[str, Any] | None = None,
    exam_id: str = "",
) -> QuestionsOnlyBundle:
    """
    Combine gradeable questions and practice-only questions into one bundle.

    The final JSON can contain both, while later TeX/reference generation
    filters practice-only questions out of grading.
    """
    gradeable = (
        gradeable_bundle
        if isinstance(gradeable_bundle, QuestionsOnlyBundle)
        else QuestionsOnlyBundle.model_validate(gradeable_bundle)
    )

    if practice_bundle is None:
        return gradeable

    practice = (
        practice_bundle
        if isinstance(practice_bundle, QuestionsOnlyBundle)
        else QuestionsOnlyBundle.model_validate(practice_bundle)
    )

    existing_ids = {int(q.question_id) for q in gradeable.questions}
    combined_questions = list(gradeable.questions)

    for q in practice.questions:
        qid = int(q.question_id)

        # If the AI accidentally re-extracts a gradeable question from the
        # practice section, do not duplicate it.
        if qid in existing_ids:
            gradeable.warnings.append(
                f"Skipped duplicate practice question_id {qid}; already exists in gradeable section."
            )
            continue

        combined_questions.append(q)

    combined_questions.sort(key=lambda q: int(q.question_id))

    gradeable.questions = combined_questions
    gradeable.exam_id = gradeable.exam_id or practice.exam_id or exam_id
    gradeable.exam_title = gradeable.exam_title or practice.exam_title

    gradeable.warnings = list(
        dict.fromkeys(
            [
                *(gradeable.warnings or []),
                *(practice.warnings or []),
            ]
        )
    )

    gradeable.structure_corrections = [
        *(gradeable.structure_corrections or []),
        *(practice.structure_corrections or []),
    ]

    gradeable.source_names = list(
        dict.fromkeys(
            [
                *(gradeable.source_names or []),
                *(practice.source_names or []),
            ]
        )
    )

    return gradeable


def _norm_text_for_alignment(text: str) -> str:
    text = str(text or "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("≤", r"\le").replace("≥", r"\ge")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _compact_math(text: str) -> str:
    text = _norm_text_for_alignment(text)
    text = re.sub(r"\\mathbb\{r\}", "r", text)
    text = re.sub(r"\\mathbb\{n\}", "n", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def _tokens_for_alignment(text: str) -> set[str]:
    text = _norm_text_for_alignment(text)

    # Keep Hebrew, English letters, digits, and common math command words.
    raw = re.findall(r"[א-תA-Za-z0-9_]+|\\[A-Za-z]+", text)
    out: set[str] = set()

    for tok in raw:
        tok = tok.strip().lower()
        if not tok or len(tok) <= 1:
            continue
        if tok in _HEBREW_STOPWORDS:
            continue
        out.add(tok)

    return out


def _latex_math_chunks(text: str) -> set[str]:
    text = str(text or "")
    chunks: set[str] = set()

    # Inline/display math chunks.
    patterns = [
        r"\$([^$]+)\$",
        r"\\\((.*?)\\\)",
        r"\\\[(.*?)\\\]",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.DOTALL):
            chunk = _compact_math(m.group(1))
            if len(chunk) >= 3:
                chunks.add(chunk)

    # Also add common raw math-looking expressions even when not wrapped.
    for m in re.finditer(r"\|[^|\n]{1,80}\|", text):
        chunk = _compact_math(m.group(0))
        if len(chunk) >= 3:
            chunks.add(chunk)

    for m in re.finditer(r"\\frac\{[^{}]+\}\{[^{}]+\}", text):
        chunk = _compact_math(m.group(0))
        if len(chunk) >= 3:
            chunks.add(chunk)

    return chunks


def _canonical_abs_inner(expr: str) -> str:
    """
    Canonicalize simple absolute-value inners:
      2-x and x-2 should be considered the same inside abs().
      6-y and y-6 should also be considered the same.

    This is intentionally conservative.
    """
    e = _compact_math(expr)
    e = e.strip("|")

    # Remove wrapping braces if present.
    e = e.strip("{}")

    # For simple a-b vs b-a, abs makes them equivalent.
    m = re.fullmatch(r"([a-zA-Z0-9]+)-([a-zA-Z0-9]+)", e)
    if m:
        a, b = m.group(1), m.group(2)
        return "-".join(sorted([a, b]))

    m = re.fullmatch(r"([a-zA-Z0-9]+)\+([a-zA-Z0-9]+)", e)
    if m:
        a, b = m.group(1), m.group(2)
        return "+".join(sorted([a, b]))

    return e


def _abs_inequality_signatures(text: str) -> set[tuple[str, str, str]]:
    """
    Extract signatures like:
      |x-2| <= 5  => ("2-x", "<=", "5")
      5 >= |2-x|  => ("2-x", "<=", "5")
      1 < |y-6|   => ("6-y", ">", "1")

    Output relation is always from abs-expression perspective.
    """
    s = _norm_text_for_alignment(text)
    s = s.replace(r"\leq", r"\le").replace(r"\geq", r"\ge")

    signatures: set[tuple[str, str, str]] = set()

    rel_patterns = [
        (r"\\le|<=", "<="),
        (r"\\ge|>=", ">="),
        (r"<", "<"),
        (r">", ">"),
    ]

    rel_regex = r"(\\le|\\ge|<=|>=|<|>)"

    def norm_rel(raw: str) -> str:
        raw = raw.strip()
        if raw in {r"\le", "<="}:
            return "<="
        if raw in {r"\ge", ">="}:
            return ">="
        return raw

    def invert_rel(rel: str) -> str:
        return {
            "<": ">",
            "<=": ">=",
            ">": "<",
            ">=": "<=",
        }.get(rel, rel)

    # |expr| < number
    for m in re.finditer(r"\|([^|\n]+)\|\s*" + rel_regex + r"\s*([0-9]+(?:\.[0-9]+)?)", s):
        inner = _canonical_abs_inner(m.group(1))
        rel = norm_rel(m.group(2))
        num = m.group(3)
        signatures.add((inner, rel, num))

    # number < |expr|
    for m in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*" + rel_regex + r"\s*\|([^|\n]+)\|", s):
        num = m.group(1)
        rel = invert_rel(norm_rel(m.group(2)))
        inner = _canonical_abs_inner(m.group(3))
        signatures.add((inner, rel, num))

    return signatures


def _answer_part_text(ap: Any) -> str:
    return "\n".join(
        [
            str(getattr(ap, "question_hint", "") or ""),
            str(getattr(ap, "official_solution", "") or ""),
            str(getattr(ap, "expected_answer", "") or ""),
            str(getattr(ap, "grading_instructions", "") or ""),
        ]
    ).strip()


def _question_part_text(qp: Any) -> str:
    return "\n".join(
        [
            str(getattr(qp, "question_text", "") or ""),
            str(getattr(qp, "required_action", "") or ""),
        ]
    ).strip()


def _topic_flags(text: str) -> set[str]:
    t = _norm_text_for_alignment(text)
    flags: set[str] = set()

    if "|" in t or "ערך מוחלט" in t:
        flags.add("absolute_value")
    if "דוגמה נגדית" in t or "counterexample" in t or "אינו נכון" in t:
        flags.add("counterexample")
    if "אינדוקציה" in t or "induction" in t:
        flags.add("induction")
    if "בינום" in t or "newton" in t or "binom" in t:
        flags.add("binomial")
    if "ברנולי" in t or "bernoulli" in t:
        flags.add("bernoulli")
    if "שטח" in t or "כיתה" in t or "מלבנית" in t:
        flags.add("area_error")
    if "סכום" in t or r"\sum" in t:
        flags.add("sum")
    if "מכפלה" in t or r"\prod" in t:
        flags.add("product")

    return flags


def _score_answer_alignment(
    *,
    question_id: int,
    part_key: str,
    question_part: Any,
    answer_question_id: int | None,
    answer_part: Any,
) -> tuple[float, list[str]]:
    """
    Return:
      (score 0..1, warnings)

    Important:
      A direct qid+part match is only a weak signal.
      Strong match requires content/math overlap.
    """
    q_text = _question_part_text(question_part)
    a_text = _answer_part_text(answer_part)

    warnings: list[str] = []

    if not a_text.strip():
        return 0.0, ["Answer candidate has no official solution text."]

    q_abs = _abs_inequality_signatures(q_text)
    a_abs = _abs_inequality_signatures(a_text)

    # Hard conflict: same absolute expression and threshold but opposite direction.
    # Example:
    #   Question: |6-y| < 1
    #   Answer:   1 < |y-6|
    for qi, qr, qn in q_abs:
        for ai, ar, an in a_abs:
            if qi == ai and qn == an and qr != ar:
                return 0.05, [
                    f"Rejected likely mismatch: same absolute expression {qi} and threshold {qn}, but inequality direction differs ({qr} vs {ar})."
                ]

    score = 0.0

    # Weak structural signal.
    if answer_question_id is not None and int(answer_question_id) == int(question_id):
        score += 0.12

    answer_pk = _part_key(
        str(getattr(answer_part, "part", "") or ""),
        str(getattr(answer_part, "part_key", "") or ""),
    )
    if answer_pk and answer_pk == part_key:
        score += 0.08

    # Strong math signature signal.
    q_chunks = _latex_math_chunks(q_text)
    a_chunks = _latex_math_chunks(a_text)

    if q_chunks and a_chunks:
        exact_overlap = q_chunks & a_chunks
        if exact_overlap:
            score += min(0.45, 0.18 * len(exact_overlap))

    # Absolute inequality exact match is very strong.
    if q_abs and a_abs:
        exact_abs = q_abs & a_abs
        if exact_abs:
            score += 0.50

        same_abs_expr = {
            qi
            for qi, _qr, _qn in q_abs
        } & {
            ai
            for ai, _ar, _an in a_abs
        }
        if same_abs_expr:
            score += 0.12

    # General token overlap.
    q_tokens = _tokens_for_alignment(q_text)
    a_tokens = _tokens_for_alignment(a_text)
    if q_tokens and a_tokens:
        jaccard = len(q_tokens & a_tokens) / max(1, len(q_tokens | a_tokens))
        score += min(0.25, jaccard * 0.75)

    # Topic overlap.
    q_topics = _topic_flags(q_text)
    a_topics = _topic_flags(a_text)
    if q_topics and a_topics:
        topic_overlap = q_topics & a_topics
        if topic_overlap:
            score += min(0.18, 0.08 * len(topic_overlap))

    # Suspicious variable mismatch in absolute-value exercises.
    if "absolute_value" in q_topics and "absolute_value" in a_topics:
        q_vars = {x for x in re.findall(r"[a-zA-Z]", _compact_math(q_text)) if x not in {"r", "n"}}
        a_vars = {x for x in re.findall(r"[a-zA-Z]", _compact_math(a_text)) if x not in {"r", "n"}}
        if q_vars and a_vars and not (q_vars & a_vars):
            score -= 0.25
            warnings.append(
                f"Suspicious variable mismatch between question variables {sorted(q_vars)} and answer variables {sorted(a_vars)}."
            )

    score = max(0.0, min(1.0, score))
    return score, warnings


def _flatten_answer_parts(ab: AnswersOnlyBundle) -> list[dict[str, Any]]:
    """
    Flatten AnswersOnlyBundle into candidates, preserving source order.
    """
    candidates: list[dict[str, Any]] = []
    order = 0

    for aq in ab.questions:
        qid = _normalize_question_id(aq.question_id)

        for ap in aq.parts:
            order += 1
            candidates.append(
                {
                    "order": order,
                    "question_id": qid,
                    "part": ap,
                    "used": False,
                }
            )

    return candidates


def _best_answer_candidate(
    *,
    question_id: int,
    part_key: str,
    question_part: Any,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float, list[str]]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_warnings: list[str] = []

    for cand in candidates:
        if cand.get("used"):
            continue

        score, warnings = _score_answer_alignment(
            question_id=question_id,
            part_key=part_key,
            question_part=question_part,
            answer_question_id=cand.get("question_id"),
            answer_part=cand.get("part"),
        )

        if score > best_score:
            best = cand
            best_score = score
            best_warnings = warnings

    return best, best_score, best_warnings


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

    Behavior:
      - If no non-submission marker exists, extract normally.
      - If a marker exists, extract the gradeable section and the practice
        section separately, then combine them into one JSON bundle.
    """
    client = client or GptClient()

    gradeable_text, practice_text, marker, section_warnings = split_submission_sections(ocr_text)

    # ------------------------------------------------------------
    # Pass 1: gradeable/submitted questions
    # ------------------------------------------------------------
    gradeable_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

This OCR section contains the submitted / gradeable questions.
Extract the questions in this section as gradeable.

Rules:
- Set is_gradeable=true.
- Set usage="gradeable".
- Set section_label="שאלות להגשה".
- Do not include answers or grading instructions.

OCR source:
-----------
{gradeable_text}
"""

    gradeable_result = client.chat_json(
        system=QUESTIONS_ONLY_SYSTEM_PROMPT,
        user=gradeable_prompt,
        schema=questions_only_bundle_schema(),
        schema_name="mathgrade_questions_only_bundle",
        strict=False,
        timeout_s=240,
    )

    gradeable_bundle = _as_questions_only_bundle(gradeable_result, exam_id=exam_id)

    for q in gradeable_bundle.questions:
        q.is_gradeable = True
        q.usage = "gradeable"
        q.section_label = q.section_label or "שאלות להגשה"

        for p in q.parts:
            p.is_gradeable = True
            p.usage = "gradeable"
            p.section_label = p.section_label or q.section_label

    # ------------------------------------------------------------
    # Pass 2: non-submission / practice-only questions
    # ------------------------------------------------------------
    practice_bundle: QuestionsOnlyBundle | None = None

    if practice_text.strip():
        practice_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}

This OCR section contains questions that are explicitly NOT for submission,
optional, practice-only, bonus, or otherwise not gradeable.

Extract these questions too, but mark them as practice-only.

Rules:
- Set is_gradeable=false for every question and every part.
- Set usage="practice_feedback_only" for every question and every part.
- Set section_label to the visible marker if possible: {marker or "non-submission"}.
- Do not include answers or grading instructions.
- Preserve visible question numbers, including starred numbers such as *.7.
- If a question has no explicit parts, represent it as one part: part="א", part_key="a".

OCR source:
-----------
{practice_text}
"""

        practice_result = client.chat_json(
            system=QUESTIONS_ONLY_SYSTEM_PROMPT,
            user=practice_prompt,
            schema=questions_only_bundle_schema(),
            schema_name="mathgrade_practice_questions_bundle",
            strict=False,
            timeout_s=240,
        )

        practice_bundle = _as_questions_only_bundle(practice_result, exam_id=exam_id)
        practice_bundle = mark_bundle_as_practice_only(
            practice_bundle,
            section_label=marker or "non-submission",
        )

    combined = combine_questions_only_bundles(
        gradeable_bundle=gradeable_bundle,
        practice_bundle=practice_bundle,
        exam_id=exam_id,
    )

    # Keep source-based marker warnings, but remove old/contradictory wording if
    # the AI generated it.
    combined.warnings = [
        w
        for w in (combined.warnings or [])
        if "הושמטו" not in str(w) and "ignored for gradeable question extraction" not in str(w)
    ]

    if section_warnings:
        combined.warnings = list(dict.fromkeys([*(combined.warnings or []), *section_warnings]))

    combined = apply_gradeable_flags_from_source(combined, ocr_text=ocr_text)

    return combined.model_dump()


def build_answers_only_bundle_from_ocr(
    *,
    ocr_text: str,
    source_name: str,
    exam_id: str,
    client: GptClient | None = None,
    questions_bundle: dict[str, Any] | QuestionsOnlyBundle | None = None,
) -> dict[str, Any]:
    """
    Input:
      OCR text from an answers-only / official-solutions teacher upload.

    Output:
      AnswersOnlyBundle as dict.
    """
    client = client or GptClient()

    locked_questions_context = ""
    if questions_bundle is not None:
        qb = (
            questions_bundle
            if isinstance(questions_bundle, QuestionsOnlyBundle)
            else QuestionsOnlyBundle.model_validate(questions_bundle)
        )
        locked_questions_context = f"""
    Locked submitted-question structure:
    -----------------------------------
    {json.dumps(qb.model_dump(), ensure_ascii=False, indent=2)}

    Important:
    - The locked question structure above is the only allowed question structure.
    - Some locked questions may have is_gradeable=false and usage="practice_feedback_only".
    - Gradeable questions are used for scoring.
    - Practice-feedback-only questions may receive answers for later student feedback, but they must not be treated as scored questions.
    - Extract answer candidates only.
    - Do not create new gradeable questions from the answer file.
    - If a visible solution appears to belong to a question that is not in the locked structure, keep it as an uncertain answer candidate or leave it unmatched.
    """

    user_prompt = f"""
    exam_id: {exam_id}
    source_name: {source_name}

    {locked_questions_context}

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


def _split_into_question_blocks(text: str) -> tuple[str, list[str]]:
    """
    Split combined questions+answers text into a shared preamble and a list of
    per-question blocks, using LaTeX ``\\section*{...}`` headers as boundaries.

    Returns (preamble, blocks). ``blocks`` is empty when fewer than two section
    headers are found, signalling the caller to fall back to a single request.
    """
    matches = list(re.finditer(r"(?m)^[ \t]*\\section\*?\s*\{", text))
    if len(matches) < 2:
        return text, []

    preamble = text[: matches[0].start()].strip()

    blocks: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start() : end].strip()
        if block:
            blocks.append(block)

    return preamble, blocks


def _group_blocks(blocks: list[str], per_chunk: int) -> list[list[str]]:
    """Group question blocks into chunks of at most ``per_chunk`` questions.

    Very short "divider" blocks (a bare section header with no body, e.g.
    "שאלות שאינן להגשה") are attached to the following block so their context is
    not stranded in a separate request.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    pending_divider: list[str] = []

    for block in blocks:
        # A divider has a header but essentially no question body.
        body = re.sub(r"(?m)^[ \t]*\\section\*?\s*\{[^}]*\}", "", block, count=1).strip()
        if not body:
            pending_divider.append(block)
            continue

        if pending_divider:
            block = "\n\n".join([*pending_divider, block])
            pending_divider = []

        current.append(block)
        if len(current) >= per_chunk:
            groups.append(current)
            current = []

    # Any trailing divider with no following question is appended so it is not lost.
    if pending_divider:
        current.extend(pending_divider)

    if current:
        groups.append(current)

    return groups


def _qa_request_one(
    *,
    client: GptClient,
    ocr_text: str,
    source_name: str,
    exam_id: str,
    timeout_s: int,
    chunk_note: str = "",
) -> dict[str, Any]:
    """Run a single questions+answers extraction request and return the raw dict."""
    user_prompt = f"""
exam_id: {exam_id}
source_name: {source_name}
{chunk_note}
OCR source:
-----------
{ocr_text}
"""

    return client.chat_json(
        system=QUESTIONS_ANSWERS_SYSTEM_PROMPT,
        user=user_prompt,
        schema=questions_answers_bundle_schema(),
        schema_name="mathgrade_questions_answers_bundle",
        strict=False,
        timeout_s=timeout_s,
    )


def _merge_questions_answers_results(
    results: list[dict[str, Any]],
    *,
    exam_id: str,
    source_name: str,
) -> dict[str, Any]:
    """Merge per-chunk QuestionsAnswersBundle dicts into one, deduping by question_id."""
    merged_questions: list[dict[str, Any]] = []
    seen_qids: set[Any] = set()
    warnings: list[str] = []
    corrections: list[Any] = []
    source_names: list[str] = []
    exam_title = ""

    for res in results:
        if not isinstance(res, dict):
            continue
        exam_title = exam_title or str(res.get("exam_title") or "")

        for q in res.get("questions") or []:
            qid = q.get("question_id") if isinstance(q, dict) else None
            if qid is not None and qid in seen_qids:
                warnings.append(
                    f"Duplicate question_id {qid} returned across chunks; kept the first occurrence."
                )
                continue
            if qid is not None:
                seen_qids.add(qid)
            merged_questions.append(q)

        warnings.extend(str(w) for w in (res.get("warnings") or []))
        corrections.extend(res.get("structure_corrections") or [])
        source_names.extend(str(s) for s in (res.get("source_names") or []))

    merged = {
        "schema_version": "questions_answers_bundle_v1",
        "exam_id": exam_id,
        "exam_title": exam_title,
        "questions": merged_questions,
        "warnings": list(dict.fromkeys(warnings)),
        "structure_corrections": corrections,
        "source_names": list(dict.fromkeys([*source_names, source_name])),
    }

    return _as_questions_answers_bundle(merged, exam_id=exam_id).model_dump()


def _exam_title_from_text(text: str) -> str:
    """Best-effort exam title from a LaTeX title command near the document top.

    Skips the preamble (``\\newcommand`` definitions often contain \\textbf) by
    searching after ``\\begin{document}`` when present.
    """
    text = text or ""
    begin = text.find(r"\begin{document}")
    head = text[begin : begin + 2500] if begin != -1 else text[:2500]

    for pat in (
        r"\\title\*?\s*\{([^}]{3,120})\}",
        r"\\(?:Large|LARGE|huge|Huge)\s*\\textbf\{([^}]{3,120})\}",
        r"\\textbf\{([^}]{3,120})\}",
    ):
        m = re.search(pat, head)
        if m:
            return m.group(1).strip()
    return ""


def build_questions_answers_bundle_from_segments(
    *,
    ocr_text: str,
    source_name: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """
    Deterministically build a QuestionsAnswersBundle from a structured teacher
    document, without calling an LLM.

    Returns None when the document has no detectable question structure, so the
    caller can fall back to the LLM-based builder.
    """
    segments = segment_document(ocr_text)
    if not segments:
        return None

    exam_title = _exam_title_from_text(ocr_text)

    questions: list[dict[str, Any]] = []
    for seg in segments:
        usage = "gradeable" if seg.is_gradeable else "practice_feedback_only"
        parts: list[dict[str, Any]] = []
        for p in seg.parts:
            parts.append(
                {
                    "part": p.label or p.part_key,
                    "part_key": p.part_key,
                    "question_text": p.question_text,
                    "required_action": "",
                    "official_solution": p.official_solution,
                    "expected_answer": p.expected_answer,
                    "grading_instructions": "",
                    "max_points": None,
                    "is_gradeable": seg.is_gradeable,
                    "usage": usage,
                    "section_label": seg.section_label,
                    "review_status": "ok",
                    "confidence": None,
                    "warnings": [],
                }
            )

        questions.append(
            {
                "question_id": seg.question_id,
                "is_gradeable": seg.is_gradeable,
                "usage": usage,
                "section_label": seg.section_label,
                "parts": parts,
            }
        )

    bundle = {
        "schema_version": "questions_answers_bundle_v1",
        "exam_id": exam_id,
        "exam_title": exam_title,
        "questions": questions,
        "warnings": [
            "Structure detected deterministically from the uploaded document "
            "(question/part segmentation), not by the language model."
        ],
        "structure_corrections": [],
        "source_names": [source_name] if source_name else [],
    }

    return _as_questions_answers_bundle(bundle, exam_id=exam_id).model_dump()


def _normalize_qa_bundle_dict(bundle: dict[str, Any], *, exam_id: str) -> dict[str, Any]:
    """
    Repair a model-produced QuestionsAnswersBundle so it is structurally sane:

      * collapse duplicate question_ids, merging their parts;
      * for a question that has content but an empty parts list, synthesise a
        single part so the question is never partless.

    This defends against models that emit one object per part (repeating the
    question_id with empty parts) or otherwise mangle the structure.
    """
    raw_questions = bundle.get("questions") or []
    merged: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []

    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        parts = [p for p in (q.get("parts") or []) if isinstance(p, dict)]

        # A partless question that still carries text at the question level:
        # lift that text into a single implicit part so it is not lost.
        if not parts:
            question_level_text = " ".join(
                str(q.get(k) or "")
                for k in ("question_text", "official_solution", "expected_answer")
            ).strip()
            if question_level_text:
                parts = [
                    {
                        "part": q.get("part") or "",
                        "part_key": normalize_part(str(q.get("part") or "")) or "a",
                        "question_text": q.get("question_text") or "",
                        "official_solution": q.get("official_solution") or "",
                        "expected_answer": q.get("expected_answer") or "",
                    }
                ]

        if qid in merged:
            existing_keys = {
                (p.get("part_key") or p.get("part") or "") for p in merged[qid]["parts"]
            }
            for p in parts:
                key = p.get("part_key") or p.get("part") or ""
                if key and key in existing_keys:
                    continue
                merged[qid]["parts"].append(p)
                existing_keys.add(key)
        else:
            merged[qid] = {**q, "parts": list(parts)}
            order.append(qid)

    bundle = {**bundle, "questions": [merged[qid] for qid in order]}
    return _as_questions_answers_bundle(bundle, exam_id=exam_id).model_dump()


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

    Behavior:
      1. First try deterministic segmentation of the document into questions and
         parts. This is robust across many teacher formats (LaTeX section/
         subsection, Hebrew/English headers, bare numbering) and does not depend
         on the language model getting the structure right. Set
         MATHGRADE_QA_DISABLE_SEGMENTER=1 to skip it.
      2. If the document has no detectable structure, fall back to the LLM
         builder. A full solution file can produce a very large JSON bundle;
         emitting it in one request risks the model's output-token limit and
         makes some models loop. So when the text splits into per-question
         blocks we process a few questions per request and merge; otherwise we
         issue a single request. Model output is normalised to collapse
         duplicate question_ids and guarantee non-empty parts.
    """
    segmenter_disabled = os.getenv("MATHGRADE_QA_DISABLE_SEGMENTER", "0").strip().lower() in {"1", "true", "yes", "on"}

    if not segmenter_disabled:
        deterministic = build_questions_answers_bundle_from_segments(
            ocr_text=ocr_text,
            source_name=source_name,
            exam_id=exam_id,
        )
        if deterministic is not None:
            return deterministic

    client = client or GptClient()

    timeout_s = int(os.getenv("MATHGRADE_QA_BUILDER_TIMEOUT_S", "240"))
    per_chunk = max(1, int(os.getenv("MATHGRADE_QA_CHUNK_SIZE", "3")))
    chunking_disabled = os.getenv("MATHGRADE_QA_CHUNK_DISABLE", "0").strip().lower() in {"1", "true", "yes", "on"}

    preamble, blocks = _split_into_question_blocks(ocr_text)
    groups = [] if chunking_disabled else _group_blocks(blocks, per_chunk)

    # Fall back to a single request when we cannot usefully chunk.
    if len(groups) < 2:
        result = _qa_request_one(
            client=client,
            ocr_text=ocr_text,
            source_name=source_name,
            exam_id=exam_id,
            timeout_s=timeout_s,
        )
        return _normalize_qa_bundle_dict(result, exam_id=exam_id)

    results: list[dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        chunk_text = "\n\n".join([preamble, *group]).strip() if preamble else "\n\n".join(group)
        chunk_note = (
            f"This is part {idx} of {len(groups)} of one exam. Extract ONLY the "
            f"questions present in the OCR source below; do not invent questions "
            f"from other parts of the exam."
        )
        results.append(
            _qa_request_one(
                client=client,
                ocr_text=chunk_text,
                source_name=source_name,
                exam_id=exam_id,
                timeout_s=timeout_s,
                chunk_note=chunk_note,
            )
        )

    merged = _merge_questions_answers_results(
        results,
        exam_id=exam_id,
        source_name=source_name,
    )
    return _normalize_qa_bundle_dict(merged, exam_id=exam_id)


def build_questions_answers_bundle_from_questions_only(
    *,
    questions_bundle: dict[str, Any] | QuestionsOnlyBundle,
    exam_id: str = "",
    source_name: str = "",
    warning: str = "",
) -> dict[str, Any]:
    """
    Convert a QuestionsOnlyBundle into a QuestionsAnswersBundle with empty answer fields.

    Used as a recovery path for combined questions+answers uploads when the heavy
    combined Q+A builder times out. The answer fallback can then generate draft
    answers from the extracted question text.
    """
    qb = (
        questions_bundle
        if isinstance(questions_bundle, QuestionsOnlyBundle)
        else QuestionsOnlyBundle.model_validate(questions_bundle)
    )

    questions: list[dict[str, Any]] = []

    for q in qb.questions:
        parts: list[dict[str, Any]] = []

        for p in q.parts:
            parts.append(
                {
                    "part": p.part or "",
                    "part_key": p.part_key or _part_key(p.part or "", p.part_key or ""),
                    "question_text": p.question_text or "",
                    "required_action": p.required_action or "",
                    "official_solution": "",
                    "expected_answer": "",
                    "grading_instructions": "",
                    "max_points": p.max_points,
                    "is_gradeable": bool(getattr(p, "is_gradeable", True)),
                    "usage": getattr(p, "usage", "gradeable"),
                    "section_label": getattr(p, "section_label", ""),
                    "review_status": "missing",
                    "confidence": p.confidence,
                    "warnings": list(
                        dict.fromkeys(
                            [
                                *(p.warnings or []),
                                "Combined questions+answers builder failed, so this part was recovered from question text only.",
                                "Answer fields require AI fallback or teacher review.",
                            ]
                        )
                    ),
                }
            )

        questions.append(
            {
                "question_id": int(q.question_id),
                "is_gradeable": bool(getattr(q, "is_gradeable", True)),
                "usage": getattr(q, "usage", "gradeable"),
                "section_label": getattr(q, "section_label", ""),
                "parts": parts,
            }
        )

    warnings = [
        *(qb.warnings or []),
        warning
        or "Combined questions+answers builder failed. Recovered questions only and left answers for fallback generation.",
    ]

    return _as_questions_answers_bundle(
        {
            "schema_version": "questions_answers_bundle_v1",
            "exam_id": exam_id or qb.exam_id,
            "exam_title": qb.exam_title or "",
            "questions": questions,
            "warnings": [w for w in warnings if str(w).strip()],
            "structure_corrections": [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in (qb.structure_corrections or [])
            ],
            "source_names": list(dict.fromkeys([*(qb.source_names or []), source_name])),
        },
        exam_id=exam_id or qb.exam_id,
    ).model_dump()


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
                    "is_gradeable": bool(getattr(p, "is_gradeable", getattr(q, "is_gradeable", True))),
                    "usage": getattr(p, "usage", getattr(q, "usage", "gradeable")),
                    "section_label": getattr(p, "section_label", getattr(q, "section_label", "")),
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
                "is_gradeable": bool(getattr(q, "is_gradeable", True)),
                "usage": getattr(q, "usage", "gradeable"),
                "section_label": getattr(q, "section_label", ""),
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

    Matching strategy:
      1. Flatten all answers into ordered candidates.
      2. For every question part, score every unused answer candidate.
      3. Attach only when the content match is strong enough.
      4. Leave uncertain/weak matches missing rather than attaching wrong answers.

    This is intentionally conservative:
      wrong answer alignment is worse than a missing answer.
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

    candidates = _flatten_answer_parts(ab)

    full_questions: list[dict[str, Any]] = []
    warnings: list[str] = []
    warnings.extend(qb.warnings or [])
    warnings.extend(ab.warnings or [])

    attached_count = 0
    missing_count = 0
    uncertain_count = 0

    for q in qb.questions:
        full_parts: list[dict[str, Any]] = []

        q_is_gradeable = bool(getattr(q, "is_gradeable", True))
        q_usage = getattr(q, "usage", "gradeable")
        q_section_label = getattr(q, "section_label", "")

        for p in q.parts:
            pk = _part_key(p.part, p.part_key)

            p_is_gradeable = bool(getattr(p, "is_gradeable", q_is_gradeable))
            p_usage = getattr(p, "usage", q_usage)
            p_section_label = getattr(p, "section_label", q_section_label)

            part_warnings = list(p.warnings or [])

            best, best_score, alignment_warnings = _best_answer_candidate(
                question_id=q.question_id,
                part_key=pk,
                question_part=p,
                candidates=candidates,
            )

            # Conservative thresholds:
            # >= 0.60 attach
            # 0.45-0.59 mention possible candidate but do not attach
            # < 0.45 ignore
            attach = best is not None and best_score >= 0.60

            if attach:
                ap = best["part"]
                best["used"] = True
                attached_count += 1

                part_warnings.extend(ap.warnings or [])
                part_warnings.extend(alignment_warnings)

                if best_score < 0.78:
                    uncertain_count += 1
                    review_status = "uncertain"
                    part_warnings.append(
                        f"Answer was attached by content alignment with moderate confidence ({best_score:.2f}). Teacher review recommended."
                    )
                else:
                    review_status = ap.review_status or "ok"

                official_solution = ap.official_solution
                expected_answer = ap.expected_answer
                grading_instructions = ap.grading_instructions
                confidence = best_score
                answer_source = ",".join(ab.source_names or [])

            else:
                missing_count += 1
                official_solution = ""
                expected_answer = ""
                grading_instructions = ""
                review_status = "missing"
                confidence = p.confidence
                answer_source = ""

                if best is not None and best_score >= 0.45:
                    part_warnings.append(
                        f"A possible answer candidate was found with weak confidence ({best_score:.2f}) but was not attached."
                    )
                    part_warnings.extend(alignment_warnings)
                else:
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
                    "is_gradeable": p_is_gradeable,
                    "usage": p_usage,
                    "section_label": p_section_label,
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
                "is_gradeable": q_is_gradeable,
                "usage": q_usage,
                "section_label": q_section_label,
                "parts": full_parts,
            }
        )

    unused_candidates = [c for c in candidates if not c.get("used")]
    for cand in unused_candidates:
        ap = cand.get("part")
        preview = _answer_part_text(ap)
        preview = re.sub(r"\s+", " ", preview).strip()[:180]
        warnings.append(
            f"Unused answer candidate #{cand.get('order')}: {preview}"
        )

    total_parts = attached_count + missing_count
    missing_rate = (missing_count / total_parts) if total_parts else 1.0

    warnings.append(
        f"Answer alignment summary: attached={attached_count}, missing={missing_count}, uncertain={uncertain_count}, unused_answers={len(unused_candidates)}, missing_rate={missing_rate:.2f}"
    )

    if attached_count == 0:
        warnings.append("No answers were aligned. This should be treated as questions-only, not a gradeable full solution.")

    if missing_rate > 0.35:
        warnings.append(
            "High missing-answer rate. This full solution should be considered a draft and not automatically gradeable."
        )

    if uncertain_count:
        warnings.append(
            f"{uncertain_count} attached answers have only moderate alignment confidence and should be reviewed."
        )

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