from __future__ import annotations

from typing import Any, Dict


def grading_response_schema() -> Dict[str, Any]:
    """JSON Schema for model output (Option B).

    We keep the original fields for compatibility with the feedback PDF renderer,
    and add richer feedback signals:
      - mismatch detection (student solved a different object)
      - common error tags
      - a single concrete "next step" suggestion
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qid": {"type": "string"},
            "max_points": {"type": "number"},
            "score": {"type": "number"},
            "summary": {"type": "string"},
            "what_was_correct": {"type": "array", "items": {"type": "string"}},
            "main_mistakes": {"type": "array", "items": {"type": "string"}},
            "how_to_improve": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "mismatch": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_mismatch": {"type": "boolean"},
                    "reference_target": {"type": "string"},
                    "student_target": {"type": "string"},
                    "explanation_he": {"type": "string"},
                },
                "required": [
                    "is_mismatch",
                    "reference_target",
                    "student_target",
                    "explanation_he",
                ],
            },
            "common_errors_detected": {"type": "array", "items": {"type": "string"}},
            "suggested_next_step_he": {"type": "string"},
        },
        "required": [
            "qid",
            "max_points",
            "score",
            "summary",
            "what_was_correct",
            "main_mistakes",
            "how_to_improve",
            "confidence",
            "mismatch",
            "common_errors_detected",
            "suggested_next_step_he",
        ],
    }
