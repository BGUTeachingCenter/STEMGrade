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
from typing import List

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
    system = load_grading_prompt()

    use_ai_input = os.getenv("MATHGRADE_USE_AI_INPUT", "0") == "1"

    graded: List[QuestionGrade] = []

    for it in items:
        payload_file = it["payload_file"]
        payload_path = payload_dir / payload_file
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

        # Stable QID: do NOT trust model for this
        qid = payload.get("qid") or payload.get("question_id") or payload_path.stem

        # Decide what we send to the model
        to_send = payload.get("ai_input") if use_ai_input else payload
        if not to_send:
            to_send = payload

        # Determine max_points from payload (never trust model)
        max_points = float(to_send.get("max_points") or payload.get("max_points") or 0.0)
        if not max_points:
            max_points = float((payload.get("rubric", {}) or {}).get("score_max") or 0.0)

        # Last resort: infer from any reference text
        if not max_points:
            ref_block = payload.get("reference", {}) or {}
            combined_text = ref_block.get("text") or ""
            solution_text = ref_block.get("solution_text") or ""
            max_points = infer_max_points(combined_text or solution_text, default_max=0.0)

        # Send ONLY the chosen object
        user = json.dumps(to_send, ensure_ascii=False, indent=2)

        resp = client.chat_json(system=system, user=user, schema=schema, temperature=0.2)

        score = float(resp.get("score", 0.0))
        if max_points > 0:
            score = max(0.0, min(score, max_points))

        confidence = max(0.0, min(float(resp.get("confidence", 0.0)), 1.0))

        graded.append(
            QuestionGrade(
                qid=qid,
                max_points=max_points,
                score=score,
                summary=str(resp.get("summary", "")),
                what_was_correct=list(resp.get("what_was_correct") or []),
                main_mistakes=list(resp.get("main_mistakes") or []),
                how_to_improve=list(resp.get("how_to_improve") or []),
                mismatch=dict(resp.get("mismatch") or {}),
                common_errors_detected=list(resp.get("common_errors_detected") or []),
                suggested_next_step_he=str(resp.get("suggested_next_step_he") or ""),
                confidence=confidence,
            )
        )

    total_score = sum(q.score for q in graded)
    total_max = sum(q.max_points for q in graded if q.max_points)

    bundle = BundleGrades(total_score=total_score, total_max=total_max, question_grades=graded)

    grades_json = out_dir / "grades.json"
    grades_json.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return grades_json
