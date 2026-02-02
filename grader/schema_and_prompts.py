# grader/schema_and_prompts.py
from __future__ import annotations

from typing import Any

GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assumptions": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "question": {"type": "string"},
                    "student_answer": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "is_correct": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "feedback_he": {"type": "string"},
                    "notes_for_teacher": {"type": "string"},
                },
                "required": [
                    "title",
                    "question",
                    "student_answer",
                    "score",
                    "is_correct",
                    "errors",
                    "feedback_he",
                    "notes_for_teacher",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assumptions", "items"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "את/ה בודק/ת עבודות במתמטיקה.\n"
    "חייב/ת להשיב בעברית בלבד.\n"
    "החזר/י JSON בלבד בהתאם לסכימה שסופקה (ללא Markdown, ללא בלוקים, ללא טקסט נוסף).\n"
    "אל תשתמש/י במילות דמה כמו RESULT_1.\n"
    "השתמש/י במילה 'התלמיד/ה'.\n"
    "ודא/י שהמשוב קצר, אנושי ומעשי.\n"
    "אם אין רובריקה מפורטת, הנח/י חלוקה שווה בין הסעיפים וציין/ני זאת בשדה assumptions.\n"
    "\n"
    "כללי תצוגת מתמטיקה חשובים:\n"
    "- אל תשתמש/י בסימנים יוניקודיים מתמטיים (כמו Σ, ∑, ∫, Δ, ε, π, ≤, ≥, ∈ וכו').\n"
    "- במקום זה כתוב/כתבי LaTeX בלבד: \\\\sum, \\\\int, \\\\Delta, \\\\varepsilon, \\\\pi, \\\\le, \\\\ge, \\\\in.\n"
    "- כל ביטוי מתמטי יש לעטוף ב-\\\\( ... \\\\).\n"
    "\n"
    "כללי JSON קריטיים:\n"
    "- בכל מחרוזת JSON: חייבים לברוח backslash, כלומר כתוב \\\\ במקום \\.\n"
    "- אם מופיע בתוכן הרצף backslash ואז האות u, כתוב \\\\u (כלומר backslash כפול ואז u).\n"
)

USER_PROMPT_TEMPLATE = (
    "בדוק/י את שיעורי הבית והענק/י ציון ומשוב לכל סעיף/תת-סעיף שניתן לזהות בקובץ.\n"
    "בכל item החזר/י:\n"
    "- title: שם הסעיף\n"
    "- question: נוסח השאלה כפי שמופיע בקובץ\n"
    "- student_answer: תשובת התלמיד/ה (בקיצור אם ארוך)\n"
    "- score: 0-100 לפי איכות ונכונות\n"
    "- feedback_he: משוב בעברית שכולל גם התייחסות לשאלה וגם לתשובת התלמיד/ה\n"
    "- errors: נקודות קצרות לתיקון\n"
    "- notes_for_teacher: הערות קצרות למורה (רובריקה/שיקולי ניקוד)\n\n"
    "חשוב: במתמטיקה כתוב/כתבי LaTeX בלבד ולא סימני Unicode.\n"
    "להלן ה-LaTeX כפי שהוגש:\n"
    "{latex}\n"
)
