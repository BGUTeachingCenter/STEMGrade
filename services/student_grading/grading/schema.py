from __future__ import annotations

from typing import Any, Dict


def grading_response_schema() -> Dict[str, Any]:
    """JSON schema for the model's response.

    We keep this strict (additionalProperties=False) to prevent the model from
    drifting. If you add fields, add them here too.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qid": {"type": "string"},
                        "max_points": {
                "type": "number",
            },
            "score": {
                "type": "number",
            },
            "correctness_level": {
                "type": "string",
                "enum": [
                    "correct",
                    "mostly_correct",
                    "partially_correct",
                    "needs_work",
                    "not_answered",
                ],
                "description": (
                    "Qualitative evaluation of the submitted "
                    "answer. This is not an official numeric grade."
                ),
            },
            "summary": {
                "type": "string",
            },
            "what_was_correct": {"type": "array", "items": {"type": "string"}},
            "main_mistakes": {"type": "array", "items": {"type": "string"}},
            "how_to_improve": {"type": "array", "items": {"type": "string"}},

            # NEW: force evidence-based feedback (short quotes)
            "evidence_correct": {"type": "array", "items": {"type": "string"}},
            "evidence_mistakes": {"type": "array", "items": {"type": "string"}},

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
                "required": ["is_mismatch", "reference_target", "student_target", "explanation_he"],
            },
            "common_errors_detected": {"type": "array", "items": {"type": "string"}},
            "suggested_next_step_he": {"type": "string"},
        },
        "required": [
            "qid",
            "max_points",
            "score",
            "correctness_level",
            "summary",
            "what_was_correct",
            "main_mistakes",
            "how_to_improve",
            "evidence_correct",
            "evidence_mistakes",
            "confidence",
            "mismatch",
            "common_errors_detected",
            "suggested_next_step_he",
        ],
    }
