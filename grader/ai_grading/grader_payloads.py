from __future__ import annotations

"""AI grading from prepared JSON payloads.

Pipeline:
  payloads/*.json + manifest.json -> grades.json

This makes grading reproducible: you can inspect or tweak the payloads before
the model sees them.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

from .grader import BundleGrades, QuestionGrade, infer_max_points
from .ollama_client import OllamaClient
from .prompting import load_grading_prompt
from .schema import grading_response_schema


def grade_payload_manifest(
    *,
    manifest_json: Path,
    out_dir: Path,
    ollama_base_url: str | None = None,
    model: str | None = None,
) -> Path:
    """Read manifest.json and grade each payload.

    Returns grades.json path.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    if ollama_base_url is None:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    payload_dir = manifest_json.parent / "payloads"

    items = manifest.get("items", [])
    if not items:
        raise RuntimeError("Manifest has no items to grade.")

    client = OllamaClient(base_url=ollama_base_url, model=model)
    schema = grading_response_schema()

    # Load human-readable grading instructions from a txt file next to this module.
    system = load_grading_prompt()

    graded: List[QuestionGrade] = []

    for it in items:
        payload_file = it["payload_file"]
        payload_path = payload_dir / payload_file
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

        # Accept both the old payload shape (reference.text) and the new one
        # (reference.solution_text) to keep backward compatibility.
        qid = payload.get("question_id") or payload.get("qid") or payload_path.stem
        ref_block = payload.get("reference", {}) or {}
        ref_text = ref_block.get("solution_text") or ref_block.get("text") or ""
        student_block = payload.get("student", {}) or {}
        student_latex = student_block.get("latex_raw", "")

        max_points = infer_max_points(ref_text, default_max=0.0)

        prompt_payload = {
            "question_id": qid,
            "reference": {
                "question_text": (ref_block.get("question_text") or ""),
                "solution_text": ref_text,
            },
            "student": {
                "latex_raw": student_latex,
                "latex_clean": student_block.get("latex_clean", ""),
            },
            "rubric": {
                "score_max": max_points,
                "key_points": (payload.get("rubric", {}) or {}).get("key_points", [])
                or [],
            },
        }

        user = json.dumps(prompt_payload, ensure_ascii=False, indent=2)

        resp = client.chat_json(system=system, user=user, schema=schema, temperature=0.2)

        score = float(resp["score"])
        if max_points > 0:
            score = max(0.0, min(score, max_points))
        conf = max(0.0, min(float(resp["confidence"]), 1.0))

        graded.append(
            QuestionGrade(
                qid=resp["qid"],
                max_points=float(resp["max_points"]),
                score=score,
                summary=resp["summary"],
                what_was_correct=list(resp["what_was_correct"]),
                main_mistakes=list(resp["main_mistakes"]),
                how_to_improve=list(resp["how_to_improve"]),
                mismatch=dict(resp.get("mismatch") or {}),
                common_errors_detected=list(resp.get("common_errors_detected") or []),
                suggested_next_step_he=str(resp.get("suggested_next_step_he") or ""),
                confidence=conf,
            )
        )

    total_score = sum(q.score for q in graded)
    total_max = sum(q.max_points for q in graded if q.max_points)
    bundle = BundleGrades(total_score=total_score, total_max=total_max, question_grades=graded)

    grades_json = out_dir / "grades.json"
    grades_json.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return grades_json
