from __future__ import annotations

"""AI grading from prepared JSON payloads.

Pipeline:
  payloads/*.json + manifest.json -> grades.json

This makes grading reproducible: you can inspect or tweak the payloads before
the model sees them.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Protocol, Any, Dict

from .grader import BundleGrades, QuestionGrade, infer_max_points

from core.ai_clients.ollama_client import OllamaClient
from core.ai_clients.google_client import GoogleClient
from core.ai_clients.gpt_client import GptClient

from .prompting import load_grading_prompt
from .schema import grading_response_schema


_WS_RE = re.compile(r"\s+")


_CORRECTNESS_LEVELS = {
    "correct",
    "mostly_correct",
    "partially_correct",
    "needs_work",
    "not_answered",
}


_CORRECTNESS_LEVEL_INSTRUCTIONS = """
In addition to the pedagogical feedback, classify the student's
answer using exactly one correctness_level:

- correct:
  The mathematical answer is correct and sufficiently complete.
  Tiny wording or presentation issues may remain.

- mostly_correct:
  The main method and conclusion are correct, but there is a minor
  omission, notation issue, missing justification, or presentation
  requirement.

- partially_correct:
  The answer contains meaningful correct reasoning, but also has a
  significant error, incomplete argument, or incorrect conclusion.

- needs_work:
  The central method or conclusion is incorrect, the answer addresses
  the wrong target, or very little usable progress was made.

- not_answered:
  There is no meaningful submitted answer.

This is a qualitative learning indicator, not an official numeric
grade. Return only one of the exact schema values.
""".strip()


_EVIDENCE_AND_VISUAL_AUDIT_INSTRUCTIONS = """
OCR AND VISUAL-EVIDENCE RULES:

The student answer may be an OCR transcription of handwritten work.
The input can include:
- student.visual_elements: descriptions of number lines, graphs, diagrams,
  tables, plotted points, shading, or other visual mathematical work;
- student.visual_capture_status: complete, partial, none, or uncertain;
- student.ocr_needs_review and student.ocr_warnings.

Mandatory grading rules:
1. Evaluate written algebra and visual_elements together. A correct number line,
   graph, table, or diagram is part of the submitted solution.
2. Never claim that a drawing, graph, number line, table, or diagram is missing
   merely because it is absent from latex_raw. Only make that claim when
   visual_capture_status is complete or none and the captured visual evidence
   supports the claim.
3. When visual_capture_status is partial or uncertain, state the uncertainty
   instead of asserting that the student omitted a visual step.
4. Treat mathematically equivalent representations as equivalent: interval
   notation, a compound inequality, set notation, and a correctly drawn number
   line may express the same solution set. Check endpoint values and whether
   endpoints are open or closed before criticizing the representation.
5. Every negative claim must be grounded in the submitted evidence. Keep
   main_mistakes and evidence_mistakes index-aligned whenever possible:
   main_mistakes[i] should be supported by evidence_mistakes[i]. Do not invent
   a quote. For a genuine omission, describe the exact missing requirement and
   leave the corresponding evidence quote empty rather than fabricating one.
6. If the OCR is uncertain and the answer could reasonably be correct, lower
   confidence and recommend review; do not turn OCR uncertainty into a student
   mistake.
""".strip()


_GRADE_VERIFIER_INSTRUCTIONS = """
You are the final grading auditor. You receive the original grading input and a
candidate grading response. Return a corrected response using the exact same
JSON schema.

Audit every negative statement before preserving it:
- Is it directly supported by the student's written or visual evidence?
- Does it contradict a visible graph, number line, diagram, table, or endpoint?
- Did the candidate mistake an equivalent representation for a missing step?
- Did OCR uncertainty get incorrectly blamed on the student?
- Are score, correctness_level, summary, mistakes, and positive feedback
  mutually consistent?

Remove unsupported criticism. Preserve valid criticism. Mention uncertainty
when the source is uncertain. Return JSON only and keep all student-facing text
in Hebrew.
""".strip()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_verify_grade(
    response: dict[str, Any],
    model_input: dict[str, Any],
) -> bool:
    if not _env_bool("MATHGRADE_ENABLE_GRADE_VERIFIER", "1"):
        return False

    if _env_bool("MATHGRADE_VERIFY_EVERY_ANSWER", "0"):
        return True

    mistakes = response.get("main_mistakes") or []
    mismatch = response.get("mismatch") or {}
    level = str(
        response.get("correctness_level") or ""
    ).strip().lower()

    try:
        confidence = float(response.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0

    student = model_input.get("student") or {}
    visual_elements = student.get("visual_elements") or []
    visual_status = str(
        student.get("visual_capture_status") or ""
    ).strip().lower()
    ocr_needs_review = bool(
        student.get("ocr_needs_review")
    )

    return bool(
        mistakes
        or mismatch.get("is_mismatch")
        or level in {
            "partially_correct",
            "needs_work",
            "not_answered",
        }
        or confidence < 0.88
        or (visual_elements and level != "correct")
        or visual_status in {"partial", "uncertain"}
        or ocr_needs_review
    )


def _normalize_correctness_level(
    value: Any,
    *,
    has_student: bool,
    score: float,
    max_points: float,
    what_was_correct: list[str],
    main_mistakes: list[str],
    mismatch: dict[str, Any],
    common_errors: list[str],
) -> str:
    """
    Validate the model classification and provide a deterministic
    fallback for older models or malformed responses.
    """
    raw = (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "fully_correct": "correct",
        "mostly": "mostly_correct",
        "partial": "partially_correct",
        "partially": "partially_correct",
        "incorrect": "needs_work",
        "wrong": "needs_work",
        "no_submission": "not_answered",
        "missing": "not_answered",
    }

    raw = aliases.get(
        raw,
        raw,
    )

    if raw in _CORRECTNESS_LEVELS:
        return raw

    if (
        not has_student
        or "no_submission" in common_errors
    ):
        return "not_answered"

    if (
        bool(
            mismatch.get(
                "is_mismatch"
            )
        )
        or "irrelevant_solution"
        in common_errors
    ):
        return "needs_work"

    if max_points > 0:
        ratio = score / max_points

        if ratio >= 0.9:
            return "correct"

        if ratio >= 0.7:
            return "mostly_correct"

        if ratio >= 0.35:
            return "partially_correct"

        return "needs_work"

    if not main_mistakes:
        return "correct"

    if (
        what_was_correct
        and len(main_mistakes) <= 1
    ):
        return "mostly_correct"

    if what_was_correct:
        return "partially_correct"

    return "needs_work"

_FEEDBACK_ONLY_SYSTEM_SUFFIX = """
SCORING MODE: FEEDBACK ONLY.

No numeric point values were configured for this question.
Evaluate correctness and quality, but do not discuss or infer a numeric score.

For schema compatibility, return 0 in the score field. Never mention that:
- the score is zero;
- the maximum score is zero;
- points were not supplied or configured;
- the score field exists only for technical reasons.

The summary and feedback must discuss only the student's mathematical work.
""".strip()


def _is_scoring_configuration_segment(
    value: Any,
) -> bool:
    text = str(
        value or ""
    ).strip().lower()

    if not text:
        return False

    mentions_score = any(
        term in text
        for term in (
            "score",
            "points",
            "max_points",
            "ציון",
            "ניקוד",
            "נקודות",
        )
    )

    mentions_configuration = any(
        term in text
        for term in (
            "zero",
            "0",
            "maximum",
            "configured",
            "provided",
            "supplied",
            "אפס",
            "מקסימום",
            "מרבי",
            "הוגדר",
            "סופק",
        )
    )

    return (
        mentions_score
        and mentions_configuration
    )


def _remove_scoring_configuration_language(
    value: Any,
) -> str:
    text = str(
        value or ""
    ).strip()

    if not text:
        return ""

    segments = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    kept = [
        segment.strip()
        for segment in segments
        if segment.strip()
        and not (
            _is_scoring_configuration_segment(
                segment
            )
        )
    ]

    return " ".join(
        kept
    ).strip()


def _clean_feedback_list(
    values: Any,
) -> list[str]:
    if not isinstance(
        values,
        list,
    ):
        values = (
            [values]
            if values not in (
                None,
                "",
            )
            else []
        )

    cleaned: list[str] = []

    for value in values:
        text = (
            _remove_scoring_configuration_language(
                value
            )
        )

        if text:
            cleaned.append(text)

    return cleaned

def _schema_for_google(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a normal JSON Schema (draft-ish) into something Gemini AI Studio accepts
    for generation_config.response_schema.

    Gemini rejects keys like: additionalProperties, $schema, title, default, definitions, $defs.
    It also doesn't like some JSON Schema constructs; we do a conservative prune.
    """

    DROP_KEYS = {
        "$schema",
        "title",
        "default",
        "examples",
        "definitions",
        "$defs",
        "additionalProperties",  # <- the one that broke you
    }

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k in DROP_KEYS:
                    continue

                # Gemini supports: type, properties, required, items, enum, description
                # It may accept: min/max, minItems, maxItems, pattern (usually ok)
                # We'll keep most validation keys except the known bad ones above.
                out[k] = walk(v)

            # If schema says "type": "object" but has no properties, Gemini can be picky.
            # Keep it as-is; your schema likely has properties.
            return out

        if isinstance(node, list):
            return [walk(x) for x in node]

        return node

    return walk(schema)



class ChatJsonClient(Protocol):
    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.15,
    ) -> dict: ...


def _clip(s: str, limit: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if len(s) <= limit else (s[:limit] + " …")


def _is_effectively_empty_latex(tex: str) -> bool:
    """Conservative: treat as empty only if there's essentially no content."""
    t = (tex or "").strip()
    if not t:
        return True
    # Strip comments
    t = re.sub(r"(?m)%.*?$", "", t)
    # Remove common no-content commands
    t = re.sub(r"\\(label|ref|cite|hfill|vspace|smallskip|medskip|bigskip)\b(\{[^}]*\})?", "", t)
    # Collapse whitespace and remove braces
    t = _WS_RE.sub("", t).replace("{", "").replace("}", "")
    return t == ""


def _make_client(
    *,
    provider: str,
):
    provider = (provider or "ollama").strip().lower()

    if provider in ("google", "gemini", "google_ai_studio", "aistudio"):
        client = GoogleClient()   # ✅ reads env only inside google_client.py
    elif provider in ("chatgpt", "openai", "gpt"):
        client = GptClient()      # ✅ reads env only inside gpt_client.py
    else:
        client = OllamaClient()   # ✅ reads env only inside ollama_client.py

    return client


def grade_payload_manifest(
    *,
    manifest_json: Path,
    out_dir: Path,
    model: str = "ollama",
    debug: bool = False,
    log_fn=None,
) -> Path:
    """Read manifest.json and grade each payload.

    Returns grades.json path.

    When debug=True, prints extra information about the grading flow.
    """

    def _log(msg: str) -> None:
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass
        if debug:
            print(msg)

    if debug:
        _log("[grade_payload_manifest] Starting")
        print(f"[grade_payload_manifest] Starting")
        print(f"[grade_payload_manifest] manifest_json={manifest_json}")
        print(f"[grade_payload_manifest] out_dir={out_dir}")
        print(f"[grade_payload_manifest] model={model}")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    payload_dir = manifest_json.parent / "payloads"

    items = manifest.get("items", [])
    if not items:
        raise RuntimeError("Manifest has no items to grade.")

    if debug:
        print(f"[grade_payload_manifest] payload_dir={payload_dir}")
        print(f"[grade_payload_manifest] items_count={len(items)}")

    client = _make_client(provider=model)
    schema = grading_response_schema()
    subject = str(manifest.get("subject") or "math")
    grading_prompt_extra = str(manifest.get("grading_prompt_extra") or "")
    system = (
            load_grading_prompt(
                subject=subject,
                extra_instructions=(
                    grading_prompt_extra
                ),
            )
            + "\n\n"
            + _CORRECTNESS_LEVEL_INSTRUCTIONS
            + "\n\n"
            + _EVIDENCE_AND_VISUAL_AUDIT_INSTRUCTIONS
    )

    provider = (model or "ollama").strip().lower()
    if provider in ("google", "gemini", "google_ai_studio", "aistudio"):
        schema = _schema_for_google(schema)
        if debug:
            print(f"[grade_payload_manifest] provider={provider} -> using Gemini-safe schema")
    elif debug:
        print(f"[grade_payload_manifest] provider={provider}")

    ref_q_limit = int(os.getenv("MATHGRADE_REF_Q_LIMIT", "4500"))
    ref_sol_limit = int(os.getenv("MATHGRADE_REF_SOL_LIMIT", "6500"))
    stu_limit = int(os.getenv("MATHGRADE_STU_LIMIT", "6500"))
    stu_clean_limit = int(os.getenv("MATHGRADE_STU_CLEAN_LIMIT", "3500"))
    temperature = float(os.getenv("MATHGRADE_TEMPERATURE", "0.15"))

    if debug:
        print(
            "[grade_payload_manifest] limits="
            f"ref_q={ref_q_limit}, ref_sol={ref_sol_limit}, "
            f"stu={stu_limit}, stu_clean={stu_clean_limit}, temp={temperature}"
        )

    graded: List[QuestionGrade] = []

    for idx, it in enumerate(items, start=1):
        payload_file = it["payload_file"]
        payload_path = payload_dir / payload_file
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

        qid = payload.get("qid") or payload.get("question_id") or payload_path.stem

        if debug:
            print("-" * 80)
            _log(f"[grade_payload_manifest] item {idx}/{len(items)}")
            print(f"[grade_payload_manifest] item {idx}/{len(items)}")
            print(f"[grade_payload_manifest] payload_file={payload_file}")
            print(f"[grade_payload_manifest] payload_path={payload_path}")
            _log(f"[grade_payload_manifest] qid={qid}")
            print(f"[grade_payload_manifest] qid={qid}")

        ref_block = payload.get("reference", {}) or {}
        stu_block = payload.get("student", {}) or {}
        rubric_block = payload.get("rubric", {}) or {}

        max_points = float(payload.get("max_points") or 0.0)
        if not max_points:
            max_points = float(rubric_block.get("score_max") or 0.0)

        if not max_points:
            combined_text = ref_block.get("text") or ""
            solution_text = ref_block.get("solution_text") or ""
            max_points = infer_max_points(combined_text or solution_text, default_max=0.0)

        scoring_mode = (
            "scored"
            if max_points > 0
            else "feedback_only"
        )

        item_system = system

        if scoring_mode == "feedback_only":
            item_system = (
                f"{system}\n\n"
                f"{_FEEDBACK_ONLY_SYSTEM_SUFFIX}"
            )

        if debug:
            print(
                "[grade_payload_manifest] "
                f"max_points={max_points}, "
                f"scoring_mode={scoring_mode}"
            )

        student_raw = str(stu_block.get("latex_raw") or "")
        no_submission = ("לא לבדיקה" in student_raw) or _is_effectively_empty_latex(student_raw)

        if debug:
            print(f"[grade_payload_manifest] student_raw_len={len(student_raw)}")
            print(f"[grade_payload_manifest] no_submission={no_submission}")

        if no_submission:
            if debug:
                print(f"[grade_payload_manifest] skipping model call for {qid} (deterministic no-submission)")

            graded.append(
                QuestionGrade(
                    qid=qid,
                    max_points=max_points,
                    score=0.0,
                    correctness_level=(
                        "not_answered"
                    ),
                    summary=(
                        "אין תשובה לבדיקה. "
                        "לא הוגשה תשובה."
                    ),
                    what_was_correct=[],
                    main_mistakes=["לא הוגשה תשובה לבדיקה."],
                    how_to_improve=["להגיש פתרון מלא.", "לכתוב את שלבי הפתרון בצורה ברורה."],
                    mismatch={"is_mismatch": False, "reference_target": "", "student_target": "", "explanation_he": ""},
                    common_errors_detected=["no_submission"],
                    suggested_next_step_he="להגיש פתרון מלא לשאלה.",
                    confidence=1.0,
                    evidence_correct=[],
                    evidence_mistakes=[],
                )
            )
            continue

        model_input = {
            "question_id": qid,
            "reference": {
                "question_text": _clip(str(ref_block.get("question_text") or ""), ref_q_limit),
                "solution_text": _clip(str(ref_block.get("solution_text") or ""), ref_sol_limit),
            },
            "student": {
                "latex_raw": _clip(
                    student_raw,
                    stu_limit,
                ),
                "latex_clean": _clip(
                    str(stu_block.get("latex_clean") or ""),
                    stu_clean_limit,
                ),
                "visual_elements": [
                    _clip(str(item), 900)
                    for item in (
                        stu_block.get("visual_elements") or []
                    )[:20]
                    if str(item or "").strip()
                ],
                "visual_capture_status": str(
                    stu_block.get("visual_capture_status")
                    or "uncertain"
                ),
                "ocr_confidence": stu_block.get("ocr_confidence"),
                "ocr_needs_review": bool(
                    stu_block.get("ocr_needs_review")
                ),
                "ocr_warnings": [
                    _clip(str(item), 500)
                    for item in (
                        stu_block.get("ocr_warnings") or []
                    )[:10]
                    if str(item or "").strip()
                ],
            },
                        "grading_mode": scoring_mode,
            "rubric": {
                "score_max": (
                    float(
                        rubric_block.get(
                            "score_max"
                        )
                        or max_points
                        or 0.0
                    )
                ),
                "score_is_configured": (
                    scoring_mode == "scored"
                ),
                "instruction": (
                    "Return a numeric score."
                    if scoring_mode == "scored"
                    else (
                        "Feedback only. Do not "
                        "mention scores, points, "
                        "or missing scoring "
                        "configuration."
                    )
                ),
                "key_points": list(
                    rubric_block.get(
                        "key_points"
                    )
                    or []
                ),
            },
        }

        if debug:
            print(
                f"[grade_payload_manifest] clipped_lengths="
                f"question={len(model_input['reference']['question_text'])}, "
                f"solution={len(model_input['reference']['solution_text'])}, "
                f"student_raw={len(model_input['student']['latex_raw'])}, "
                f"student_clean={len(model_input['student']['latex_clean'])}"
            )
            print(f"[grade_payload_manifest] calling client.chat_json for qid={qid}")

        user = json.dumps(model_input, ensure_ascii=False, indent=2)
        resp = client.chat_json(
            system=item_system,
            user=user,
            schema=schema,
            temperature=temperature,
        )

        if debug:
            print(
                "[grade_payload_manifest] "
                f"model response keys={sorted(resp.keys())}"
            )

        if _should_verify_grade(resp, model_input):
            if debug:
                print(
                    "[grade_payload_manifest] "
                    f"running verifier for qid={qid}"
                )

            verifier_user = json.dumps(
                {
                    "original_input": model_input,
                    "candidate_grading": resp,
                },
                ensure_ascii=False,
                indent=2,
            )

            try:
                verified = client.chat_json(
                    system=(
                        f"{item_system}\n\n"
                        f"{_GRADE_VERIFIER_INSTRUCTIONS}"
                    ),
                    user=verifier_user,
                    schema=schema,
                    temperature=0.0,
                )

                if isinstance(verified, dict):
                    resp = verified

                    if debug:
                        print(
                            "[grade_payload_manifest] "
                            f"verifier accepted for qid={qid}"
                        )
            except Exception as verifier_error:
                if debug:
                    print(
                        "[grade_payload_manifest] verifier failed; "
                        f"keeping first response for qid={qid}: "
                        f"{verifier_error}"
                    )

        student_raw2 = (
            (payload.get("student") or {}).get("latex_raw")
            or ""
        ).strip()
        has_student = len(student_raw2) > 0

        score = float(resp.get("score", 0.0))

        summary = str(resp.get("summary", "")).strip()
        what_was_correct = list(resp.get("what_was_correct") or [])
        main_mistakes = list(resp.get("main_mistakes") or [])
        how_to_improve = list(resp.get("how_to_improve") or [])
        common_errors = list(resp.get("common_errors_detected") or [])
        mismatch = dict(resp.get("mismatch") or {})
        suggested_next = str(resp.get("suggested_next_step_he") or "").strip()
        confidence = max(
            0.0,
            min(
                float(
                    resp.get(
                        "confidence",
                        0.0,
                    )
                ),
                1.0,
            ),
        )

        if scoring_mode == "feedback_only":
            # The response schema currently requires a number, but it must
            # not become a student-visible zero score.
            score = 0.0

            summary = (
                _remove_scoring_configuration_language(
                    summary
                )
            )

            what_was_correct = (
                _clean_feedback_list(
                    what_was_correct
                )
            )

            main_mistakes = (
                _clean_feedback_list(
                    main_mistakes
                )
            )

            how_to_improve = (
                _clean_feedback_list(
                    how_to_improve
                )
            )

            suggested_next = (
                _remove_scoring_configuration_language(
                    suggested_next
                )
            )

            if not summary:
                summary = (
                    "המשוב המפורט מופיע "
                    "בסעיפים הבאים."
                )

        no_submission_lang = (
            ("לא הוגשה" in summary) or
            ("אין תשובה" in summary) or
            ("no submission" in summary.lower()) or
            ("missing answer" in summary.lower())
        )
        no_submission_tag = ("no_submission" in common_errors)

        if debug:
            print(
                f"[grade_payload_manifest] raw_score={score}, "
                f"confidence={confidence}, has_student={has_student}, "
                f"no_submission_lang={no_submission_lang}, no_submission_tag={no_submission_tag}"
            )

        if has_student and (no_submission_lang or no_submission_tag):
            if debug:
                print(f"[grade_payload_manifest] applying guardrail against false no-submission for qid={qid}")

            min_partial = float(os.getenv("MATHGRADE_MIN_SCORE_IF_SUBMITTED", "0"))
            score = max(score, min_partial)

            if not summary or no_submission_lang:
                summary = "הוגשה תשובה, אך היא אינה תואמת לשאלה/לפתרון הנדרש (ייתכן שנפתרה שאלה אחרת)."

            if not main_mistakes:
                main_mistakes = ["הפתרון מתייחס לביטוי/אינטגרל שונה מזה שבשאלה."]

            if not how_to_improve:
                how_to_improve = [
                    "קרא/י שוב את השאלה וודא/י שהאובייקט המתמטי נכון (אינטגרל/תחום/גבולות).",
                    "השווה/י לשלד הפתרון הרשמי: מה צריך להוכיח/לחשב בדיוק?"
                ]

            if not mismatch:
                mismatch = {
                    "is_mismatch": True,
                    "reference_target": (payload.get("reference") or {}).get("solution_text", "")[:500],
                    "student_target": student_raw2[:500],
                    "explanation_he": "הוגשה תשובה אך היא אינה תואמת את הדרישה; ייתכן שהסטודנט חישב משהו אחר."
                }
            else:
                mismatch.setdefault("is_mismatch", True)

            common_errors = [e for e in common_errors if e != "no_submission"]
            if "irrelevant_solution" not in common_errors:
                common_errors.append("irrelevant_solution")

            confidence = max(confidence, 0.4)

        if max_points > 0:
            score = max(
                0.0,
                min(
                    score,
                    max_points,
                ),
            )

        correctness_level = (
            _normalize_correctness_level(
                resp.get(
                    "correctness_level"
                ),
                has_student=has_student,
                score=score,
                max_points=max_points,
                what_was_correct=(
                    what_was_correct
                ),
                main_mistakes=(
                    main_mistakes
                ),
                mismatch=mismatch,
                common_errors=(
                    common_errors
                ),
            )
        )

        if debug:
            print(
                f"[grade_payload_manifest] final_score={score}, "
                f"summary_len={len(summary)}, "
                f"correct_count={len(what_was_correct)}, "
                f"mistakes_count={len(main_mistakes)}, "
                f"improve_count={len(how_to_improve)}, "
                f"errors_count={len(common_errors)}, "
                f"correctness={correctness_level}"
            )

        graded.append(
            QuestionGrade(
                qid=qid,
                max_points=max_points,
                score=score,
                correctness_level=correctness_level,
                summary=summary,
                what_was_correct=what_was_correct,
                main_mistakes=main_mistakes,
                how_to_improve=how_to_improve,
                mismatch=mismatch,
                common_errors_detected=common_errors,
                suggested_next_step_he=suggested_next,
                confidence=confidence,
                evidence_correct=list(resp.get("evidence_correct") or []),
                evidence_mistakes=list(resp.get("evidence_mistakes") or []),
            )
        )

    total_score = sum(q.score for q in graded)
    total_max = sum(q.max_points for q in graded if q.max_points)

    if debug:
        print("=" * 80)
        print(f"[grade_payload_manifest] total_score={total_score}")
        print(f"[grade_payload_manifest] total_max={total_max}")
        print(f"[grade_payload_manifest] graded_items={len(graded)}")
        _log(f"[grade_payload_manifest] graded_items={len(graded)}")

    bundle = BundleGrades(
        total_score=total_score,
        total_max=total_max,
        question_grades=graded,
    )

    bundle_dict = bundle.to_dict()

    question_grade_dicts = bundle_dict.get(
        "questions"
    )

    # Backward-compatible fallback in case an older BundleGrades
    # implementation uses the previous key.
    if not isinstance(
            question_grade_dicts,
            list,
    ):
        question_grade_dicts = bundle_dict.get(
            "question_grades"
        )

    if not isinstance(
            question_grade_dicts,
            list,
    ):
        question_grade_dicts = []

    scored_count = 0

    for index, grade_dict in enumerate(
            question_grade_dicts
    ):
        if not isinstance(
                grade_dict,
                dict,
        ):
            continue

        grade_model = (
            graded[index]
            if index < len(graded)
            else None
        )

        max_points_value = float(
            getattr(
                grade_model,
                "max_points",
                0.0,
            )
            or 0.0
        )

        if max_points_value > 0:
            grade_dict[
                "scoring_mode"
            ] = "scored"

            scored_count += 1

        else:
            grade_dict[
                "scoring_mode"
            ] = "feedback_only"

            grade_dict["score"] = None
            grade_dict[
                "max_points"
            ] = None

    if scored_count == len(graded):
        bundle_scoring_mode = "scored"

    elif scored_count:
        bundle_scoring_mode = "mixed"

    else:
        bundle_scoring_mode = (
            "feedback_only"
        )

    score_available = (
            scored_count > 0
    )

    bundle_dict["scoring"] = {
        "mode": bundle_scoring_mode,
        "score_available": (
            score_available
        ),
        "scored_part_count": (
            scored_count
        ),
        "feedback_only_part_count": (
                len(graded)
                - scored_count
        ),
    }

    if not score_available:
        bundle_dict[
            "total_score"
        ] = None

        bundle_dict[
            "total_max"
        ] = None

    grades_json = (
            out_dir
            / "grades.json"
    )

    grades_json.write_text(
        json.dumps(
            bundle_dict,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if debug:
        print(f"[grade_payload_manifest] wrote grades_json={grades_json}")

    try:
        usage_summary = {
            "provider": provider,
            "total_tokens": int(getattr(client, "total_tokens", 0) or 0),
            "prompt_tokens": int(getattr(client, "total_prompt_tokens", 0) or 0),
            "candidate_tokens": int(getattr(client, "total_candidate_tokens", 0) or 0),
        }
        usage_path = out_dir / "usage_summary.json"
        usage_path.write_text(
            json.dumps(usage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if debug:
            print(f"[grade_payload_manifest] wrote usage_summary={usage_path}")
            print(f"[grade_payload_manifest] usage={usage_summary}")
    except Exception as e:
        if debug:
            print(f"[grade_payload_manifest] failed to write usage_summary: {e}")
            _log(f"[grade_payload_manifest] failed to write usage_summary: {e}")

    if debug:
        print(f"[grade_payload_manifest] Finished successfully")
        _log(f"[grade_payload_manifest] Finished successfully")

    return grades_json
