from __future__ import annotations

from typing import Any, Dict


def grading_response_schema() -> Dict[str, Any]:
    """
    JSON Schema for model output.
    Keep it simple and strict so you can render to PDF reliably.
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
        ],
    }
