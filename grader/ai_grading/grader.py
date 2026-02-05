from __future__ import annotations

import json, hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .ollama_client import OllamaClient
from .pdf_extract import split_bundle_pdf_into_questions
from .prompting import load_grading_prompt
from .schema import grading_response_schema
from .latex_render import render_feedback_tex  # uses math normalization internally

from datetime import datetime

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def dump_payload(run_dir: Path, qkey: str, payload: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload2 = dict(payload)
    payload2["_meta"] = {
        "dumped_at": datetime.utcnow().isoformat() + "Z",
        "hashes": {
            k: sha256(v) for k, v in payload.items()
            if isinstance(v, str)
        }
    }
    (run_dir / f"{qkey}_payload.json").write_text(
        json.dumps(payload2, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )



@dataclass
class QuestionGrade:
    qid: str
    max_points: float
    score: float
    summary: str
    what_was_correct: List[str]
    main_mistakes: List[str]
    how_to_improve: List[str]
    mismatch: dict
    common_errors_detected: List[str]
    suggested_next_step_he: str
    confidence: float


@dataclass
class BundleGrades:
    total_score: float
    total_max: float
    question_grades: List[QuestionGrade]

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "total_max": self.total_max,
            "questions": [q.__dict__ for q in self.question_grades],
        }


_POINTS_RE = re.compile(r"\(\s*(\d+)\s*נקודות\s*\)")


def infer_max_points(reference_solution_text: str, default_max: float = 0.0) -> float:
    if not reference_solution_text:
        return default_max
    m = _POINTS_RE.search(reference_solution_text)
    return float(m.group(1)) if m else default_max


def grade_bundle_pdf(
    bundle_pdf: Path,
    out_dir: Path,
    ollama_base_url: str | None = None,
    model: str | None = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if ollama_base_url is None:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    questions = split_bundle_pdf_into_questions(bundle_pdf)
    if not questions:
        raise RuntimeError("Could not detect questions in bundle PDF text extraction.")

    client = OllamaClient(base_url=ollama_base_url, model=model)

    system = load_grading_prompt()
    schema = grading_response_schema()

    graded: List[QuestionGrade] = []
    for qid, qt in questions.items():
        max_points = infer_max_points(qt.reference_solution, default_max=0.0)

        payload = {
            "question_id": qid,
            "reference": {
                "question_text": "",
                "solution_text": qt.reference_solution,
            },
            "student": {
                "latex_raw": qt.student_answer,
            },
            "rubric": {
                "score_max": max_points,
                "key_points": [],
            },
        }
        # dump_payload(user_accessible_run_dir, qid, payload)

        user = json.dumps(payload["ai_input"], ensure_ascii=False, indent=2)
        resp = client.chat_json(system=system, user=user, schema=schema, temperature=0.2)

        score = float(resp.get("score", 0.0))
        if max_points > 0:
            score = max(0.0, min(score, max_points))

        confidence = float(resp.get("confidence", 0.0))
        confidence = max(0.0, min(confidence, 1.0))

        graded.append(
            QuestionGrade(
                qid=str(resp.get("qid", qid)),
                max_points=float(resp.get("max_points", max_points)),
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

    grades_tex = out_dir / "graded_feedback.tex"
    grades_tex.write_text(render_feedback_tex(bundle, bundle_pdf.name), encoding="utf-8")

    return grades_json, grades_tex
