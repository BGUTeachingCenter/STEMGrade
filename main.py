# grader.py
import json, urllib.request, pathlib

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5"

GRADE_SCHEMA = {
  "type": "object",
  "properties": {
    "total_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "score": {"type": "integer", "minimum": 0, "maximum": 100},
          "is_correct": {"type": "boolean"},
          "feedback_he": {"type": "string"},
          "notes_for_teacher": {"type": "string"},
        },
        "required": ["title","score","is_correct","feedback_he","notes_for_teacher"],
        "additionalProperties": False
      }
    }
  },
  "required": ["total_score","items"],
  "additionalProperties": False
}

SYSTEM = (
  "את/ה בודק/ת עבודות במתמטיקה.\n"
  "החזר/י JSON בלבד בהתאם לסכימה.\n"
  "אל תשתמש/י בסימני גרשיים לא-מוסברים בתוך טקסט.\n"
  "המשוב חייב להיות בעברית, ברור וקצר.\n"
)

def ollama_grade(latex_text: str) -> dict:
    prompt = (
      "בדוק/י את שיעורי הבית בלת״כ. תן/י ציון ומשוב לכל סעיף.\n"
      "אם אין רובריקה, הנח/י חלוקה שווה בין הסעיפים והסבר/י את ההנחות בהערות למורה.\n\n"
      "להלן ה-LaTeX כפי שהוגש:\n"
      f"{latex_text}"
    )

    payload = {
      "model": MODEL,
      "system": SYSTEM,
      "prompt": prompt,
      "format": GRADE_SCHEMA,
      "stream": False,
      "options": {"temperature": 0}
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
      OLLAMA_URL, data=data,
      headers={"Content-Type":"application/json"},
      method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return json.loads(out["response"])

def main():
    tex_path = pathlib.Path("012.tex")
    latex = tex_path.read_text(encoding="utf-8", errors="ignore")
    result = ollama_grade(latex)
    pathlib.Path("grade.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote grade.json")

if __name__ == "__main__":
    main()
