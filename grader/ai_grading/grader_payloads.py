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

from grader.ai_clients.ollama_client import OllamaClient
from grader.ai_clients.google_client import GoogleClient

from .prompting import load_grading_prompt
from .schema import grading_response_schema


_WS_RE = re.compile(r"\s+")

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
    else:
        client = OllamaClient()  # ✅ reads env only inside ollama_client.py

    return client


def grade_payload_manifest(
    *,
    manifest_json: Path,
    out_dir: Path,
    model: str = "ollama",
) -> Path:
    """Read manifest.json and grade each payload.

    Returns grades.json path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    payload_dir = manifest_json.parent / "payloads"

    items = manifest.get("items", [])
    if not items:
        raise RuntimeError("Manifest has no items to grade.")

    client = _make_client(provider=model)
    schema = grading_response_schema()
    system = load_grading_prompt()

    provider = (os.getenv("MATHGRADE_PROVIDER") or "ollama").strip().lower()
    if provider in ("google", "gemini", "google_ai_studio", "aistudio"):
        schema = _schema_for_google(schema)

    # Optional clipping limits to reduce LLM confusion
    ref_q_limit = int(os.getenv("MATHGRADE_REF_Q_LIMIT", "4500"))
    ref_sol_limit = int(os.getenv("MATHGRADE_REF_SOL_LIMIT", "6500"))
    stu_limit = int(os.getenv("MATHGRADE_STU_LIMIT", "6500"))
    stu_clean_limit = int(os.getenv("MATHGRADE_STU_CLEAN_LIMIT", "3500"))
    temperature = float(os.getenv("MATHGRADE_TEMPERATURE", "0.15"))

    graded: List[QuestionGrade] = []

    for it in items:
        payload_file = it["payload_file"]
        payload_path = payload_dir / payload_file
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

        # Stable QID: do NOT trust model for this
        qid = payload.get("qid") or payload.get("question_id") or payload_path.stem

        ref_block = payload.get("reference", {}) or {}
        stu_block = payload.get("student", {}) or {}
        rubric_block = payload.get("rubric", {}) or {}

        # Determine max_points from payload (never trust model)
        max_points = float(payload.get("max_points") or 0.0)
        if not max_points:
            max_points = float(rubric_block.get("score_max") or 0.0)

        # Last resort: infer from reference text
        if not max_points:
            combined_text = ref_block.get("text") or ""
            solution_text = ref_block.get("solution_text") or ""
            max_points = infer_max_points(combined_text or solution_text, default_max=0.0)

        student_raw = str(stu_block.get("latex_raw") or "")
        no_submission = ("לא לבדיקה" in student_raw) or _is_effectively_empty_latex(student_raw)

        # Deterministic no-submission (do NOT call model)
        if no_submission:
            graded.append(
                QuestionGrade(
                    qid=qid,
                    max_points=max_points,
                    score=0.0,
                    summary="אין תשובה לבדיקה. לא הוגשה תשובה.",
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

        # IMPORTANT: Always send the exact shape the prompt describes
        model_input = {
            "question_id": qid,
            "reference": {
                "question_text": _clip(str(ref_block.get("question_text") or ""), ref_q_limit),
                "solution_text": _clip(str(ref_block.get("solution_text") or ""), ref_sol_limit),
            },
            "student": {
                "latex_raw": _clip(student_raw, stu_limit),
                "latex_clean": _clip(str(stu_block.get("latex_clean") or ""), stu_clean_limit),
            },
            "rubric": {
                "score_max": float(rubric_block.get("score_max") or max_points or 0.0),
                "key_points": list(rubric_block.get("key_points") or []),
            },
        }

        user = json.dumps(model_input, ensure_ascii=False, indent=2)

        resp = client.chat_json(system=system, user=user, schema=schema, temperature=temperature)

        # ---------------------------
        # Post-process / guardrails
        # ---------------------------
        student_raw2 = ((payload.get("student") or {}).get("latex_raw") or "").strip()
        has_student = len(student_raw2) > 0

        score = float(resp.get("score", 0.0))

        summary = str(resp.get("summary", "")).strip()
        what_was_correct = list(resp.get("what_was_correct") or [])
        main_mistakes = list(resp.get("main_mistakes") or [])
        how_to_improve = list(resp.get("how_to_improve") or [])
        common_errors = list(resp.get("common_errors_detected") or [])
        mismatch = dict(resp.get("mismatch") or {})
        suggested_next = str(resp.get("suggested_next_step_he") or "").strip()
        confidence = max(0.0, min(float(resp.get("confidence", 0.0)), 1.0))

        # Detect "no submission" language
        no_submission_lang = (
            ("לא הוגשה" in summary) or
            ("אין תשובה" in summary) or
            ("no submission" in summary.lower()) or
            ("missing answer" in summary.lower())
        )
        no_submission_tag = ("no_submission" in common_errors)

        if has_student and (no_submission_lang or no_submission_tag):
            # If work exists, don't allow a bogus "missing answer" response.
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

        # Clamp score
        if max_points > 0:
            score = max(0.0, min(score, max_points))

        graded.append(
            QuestionGrade(
                qid=qid,
                max_points=max_points,
                score=score,
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

    bundle = BundleGrades(total_score=total_score, total_max=total_max, question_grades=graded)

    grades_json = out_dir / "grades.json"
    grades_json.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return grades_json
