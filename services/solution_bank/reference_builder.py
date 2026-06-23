from __future__ import annotations

import json
import re
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

        for p in q.parts:
            pk = _part_key(p.part, p.part_key)

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