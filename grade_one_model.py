import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/api/generate"

# Input / output
INPUT_TEX = Path("012.tex")
BUILD_DIR = Path("build")
CLEAN_PDF = BUILD_DIR / "012_clean.pdf"          # produced by compile_tex.py
GRADED_TEX = BUILD_DIR / "graded_report.tex"
GRADED_PDF = BUILD_DIR / "graded_report.pdf"

# Choose the model per run
MODEL = "gemma3:4b"  # change to "gemma3:4b" etc.

# Timeout for model call (seconds). Increase if needed.
OLLAMA_TIMEOUT = 420

# --- JSON schema for structured grading ---
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
    "להלן ה-LaTeX כפי שהוגש:\n"
    "{latex}\n"
)

# --- Helpers ---
def run(cmd: list[str], cwd: Path | None = None):
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(cmd))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise

def call_ollama_grade(model: str, system: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "format": GRADE_SCHEMA,     # structured output
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return json.loads(out["response"])

def compute_total(items: list[dict]) -> int:
    if not items:
        return 0
    return round(sum(int(it.get("score", 0)) for it in items) / len(items))

_tex_escape_map = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

def escape_tex(s: str) -> str:
    """Escape user/model text so it won't break LaTeX compilation."""
    if s is None:
        return ""
    out = []
    for ch in s:
        out.append(_tex_escape_map.get(ch, ch))
    # preserve newlines
    return "".join(out).replace("\n", r"\\ " + "\n")

def build_report_tex(model: str, system_prompt: str, user_prompt: str, grade: dict, total_score: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    items = grade["items"]
    assumptions = escape_tex(grade.get("assumptions", ""))

    # NOTE: We embed the already-compiled student PDF for perfect layout.
    lines = []
    lines.append(r"\documentclass[12pt]{article}" "\n")
    lines.append(r"\usepackage{fontspec}" "\n")
    lines.append(r"\usepackage{bidi}" "\n")
    lines.append(r"\setmainfont[Script=Hebrew]{Arial}" "\n")
    lines.append(r"\setRTL" "\n")
    lines.append(r"\usepackage[a4paper,margin=2cm]{geometry}" "\n")
    lines.append(r"\usepackage{longtable}" "\n")
    lines.append(r"\usepackage{hyperref}" "\n")
    lines.append(r"\usepackage{pdfpages}" "\n")
    lines.append(r"\usepackage{xcolor}" "\n")
    lines.append(r"\begin{document}" "\n\n")

    # Cover / metadata
    lines.append(r"{\Large \textbf{דו''ח בדיקה – שיעורי בית}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(rf"\textbf{{תאריך:}} {escape_tex(now)}\par" "\n")
    lines.append(rf"\textbf{{מודל:}} {escape_tex(model)}\par" "\n")
    lines.append(rf"\textbf{{ציון סופי (מחושב):}} {total_score}/100\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\textbf{הנחות:}\par" "\n")
    lines.append(assumptions + r"\par" "\n")
    lines.append(r"\vspace{0.6cm}" "\n")

    # Prompts used
    lines.append(r"\textbf{System prompt used:}\par" "\n")
    lines.append(r"\begin{quote}\ttfamily" "\n")
    lines.append(escape_tex(system_prompt) + "\n")
    lines.append(r"\end{quote}" "\n")
    lines.append(r"\textbf{User prompt used:}\par" "\n")
    lines.append(r"\begin{quote}\ttfamily" "\n")
    # user prompt can be huge; include only first 1500 chars
    lines.append(escape_tex(user_prompt[:1500]) + "\n")
    lines.append(r"\end{quote}" "\n")
    lines.append(r"\newpage" "\n")

    # Breakdown table
    lines.append(r"{\Large \textbf{פירוט ציונים ומשוב}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\begin{longtable}{|p{0.26\textwidth}|p{0.08\textwidth}|p{0.58\textwidth}|}" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{ציון} & \textbf{משוב לתלמיד/ה} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endfirsthead" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{ציון} & \textbf{משוב לתלמיד/ה} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endhead" "\n")

    for it in items:
        title = escape_tex(it.get("title", ""))
        score = int(it.get("score", 0))
        question = escape_tex(it.get("question", ""))
        student_answer = escape_tex(it.get("student_answer", ""))
        feedback = escape_tex(it.get("feedback_he", ""))
        errors = it.get("errors", [])

        err_block = ""
        if errors:
            err_lines = [escape_tex(e) for e in errors]
            err_block = r"\textbf{נקודות לתיקון:} " + r" \newline ".join(err_lines)

        cell = (
            r"\textbf{השאלה:} " + question + r"\newline "
            r"\textbf{תשובת התלמיד/ה:} " + student_answer + r"\newline "
            r"\textbf{משוב:} " + feedback
        )
        if err_block:
            cell += r"\newline " + err_block

        lines.append(f"{title} & {score} & {cell} \\\\" "\n")
        lines.append(r"\hline" "\n")

    lines.append(r"\end{longtable}" "\n")
    lines.append(r"\newpage" "\n")

    # Append original homework PDF (compiled cleanly)
    lines.append(r"{\Large \textbf{הגשה מקורית (PDF כפי שהודפס מ-LaTeX)}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\includepdf[pages=-]{012_clean.pdf}" "\n")

    lines.append(r"\end{document}" "\n")
    return "".join(lines)

def compile_report_tex():
    # compile graded_report.tex into graded_report.pdf inside BUILD_DIR
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={BUILD_DIR}", str(GRADED_TEX)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={BUILD_DIR}", str(GRADED_TEX)])

# --- Main pipeline ---
def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_TEX.exists():
        raise FileNotFoundError(f"Missing {INPUT_TEX.resolve()}")

    # 1) Compile student submission using your existing compile_tex.py
    #    This produces build/012_clean.pdf which we embed later.
    print("Step 1: Compiling student submission with compile_tex.py ...")
    run(["python", "compile_tex.py"])

    if not CLEAN_PDF.exists():
        raise RuntimeError("Expected build/012_clean.pdf was not created. Check compile_tex.py output.")

    # 2) Grade using ONE model
    print(f"Step 2: Grading with model: {MODEL} ...")
    latex_text = INPUT_TEX.read_text(encoding="utf-8", errors="ignore")
    user_prompt = USER_PROMPT_TEMPLATE.format(latex=latex_text)
    grade = call_ollama_grade(MODEL, SYSTEM_PROMPT, user_prompt)

    # Compute total in Python (don’t trust model totals)
    total = compute_total(grade.get("items", []))

    # Save grading JSON for debugging
    (BUILD_DIR / "grade.json").write_text(json.dumps(grade, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Build LaTeX report (feedback + embedded student PDF)
    print("Step 3: Building graded_report.tex ...")
    tex_report = build_report_tex(MODEL, SYSTEM_PROMPT, user_prompt, grade, total)
    GRADED_TEX.write_text(tex_report, encoding="utf-8")

    # 4) Compile LaTeX report to final PDF
    print("Step 4: Compiling graded_report.pdf ...")
    compile_report_tex()

    if not GRADED_PDF.exists():
        raise RuntimeError("graded_report.pdf not created. Check LaTeX logs in build/.")

    print("\nDone.")
    print(f"- Clean student PDF: {CLEAN_PDF}")
    print(f"- Grading JSON      : {BUILD_DIR / 'grade.json'}")
    print(f"- Final graded PDF  : {GRADED_PDF}")

if __name__ == "__main__":
    main()
