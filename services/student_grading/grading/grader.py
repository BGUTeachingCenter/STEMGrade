from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class QuestionGrade:
    qid: str
    max_points: float
    score: float
    correctness_level: str
    summary: str
    what_was_correct: List[str]
    main_mistakes: List[str]
    how_to_improve: List[str]
    mismatch: dict
    common_errors_detected: List[str]
    suggested_next_step_he: str
    confidence: float
    evidence_correct: List[str]
    evidence_mistakes: List[str]


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