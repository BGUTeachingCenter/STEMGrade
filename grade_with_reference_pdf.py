import json
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/api/generate"

# Inputs
INPUT_TEX = Path("012.tex")
REFERENCE_PDF = Path("midtermCalc2sol.pdf")  # questions + official solutions

# Output
BUILD_DIR = Path("build")
CLEAN_PDF = BUILD_DIR / "012_clean.pdf"          # produced by compile_tex.py
GRADED_TEX = BUILD_DIR / "graded_report.tex"
GRADED_PDF = BUILD_DIR / "graded_report.pdf"

# Choose the model per run
MODEL = "gemma3:4b"  # change per run

# Timeout for model call (seconds). Increase if needed.
OLLAMA_TIMEOUT = 420

# --- JSON schema for structured grading ---
# We grade against the official PDF and award points (not just %).
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
                    "qnum": {"type": "integer", "minimum": 1},
                    "part": {"type": "string"},
                    "question": {"type": "string"},
                    "student_answer": {"type": "string"},
                    "points_possible": {"type": "integer", "minimum": 0, "maximum": 100},
                    "points_awarded": {"type": "integer", "minimum": 0, "maximum": 100},
                    "is_correct": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "feedback_he": {"type": "string"},
                    "notes_for_teacher": {"type": "string"},
                },
                "required": [
                    "title",
                    "qnum",
                    "part",
                    "question",
                    "student_answer",
                    "points_possible",
                    "points_awarded",
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
    "הפתרון הרשמי המצורף הוא אמת מידה: אם תשובת התלמיד/ה שקולה/נכונה, תן/ני נקודות מלאות גם אם הניסוח שונה.\n"
    "תן/ני נקודות חלקיות כאשר יש דרך נכונה חלקית או טעויות חישוב קלות.\n"
    "אם אין רובריקה מפורטת, הנח/י חלוקת נקודות הגיונית בתוך הסעיף וציין/ני זאת בשדה assumptions.\n"
)

USER_PROMPT_TEMPLATE = (
    "בדוק/י את תשובות התלמיד/ה בכל סעיף לפי השאלות והפתרונות הרשמיים המצורפים.\n"
    "חשוב: בבוחן המקורי נכתב שרק הניסוח המדויק 'אני לא יודע/ת' מזכה ב-20% מהניקוד; ניסוחים אחרים (למשל 'לא לבדיקה') מקבלים 0.\n"
    "החזר/י items לפי הרשימה שסיפקתי (אל תוסיף/י סעיפים חדשים).\n\n"
    "לכל item:\n"
    "- title: למשל 'שאלה 2(ב)'\n"
    "- qnum, part: מספר שאלה ואות סעיף\n"
    "- question: נוסח השאלה\n"
    "- student_answer: תשובת התלמיד/ה (בקיצור אם ארוך)\n"
    "- points_possible: נקודות הסעיף\n"
    "- points_awarded: כמה נקודות לתת בפועל (0..points_possible)\n"
    "- feedback_he: משוב בעברית, קצר ומעשי\n"
    "- errors: נקודות קצרות לתיקון\n"
    "- notes_for_teacher: שיקולי ניקוד / מה נחשב נכון\n\n"
    "להלן נתוני הבדיקה לכל סעיף (שאלה, פתרון רשמי, ותשובת התלמיד/ה):\n"
    "{payload}\n"
)

# ---------- helpers ----------
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
        "format": GRADE_SCHEMA,  # structured output
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

def _clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(\n\s*\d+\s*)+$", "", s)  # trailing page numbers
    return s.strip()

def extract_reference_items(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts (qnum, part, points, question, official_solution) from the PDF.
    Notes:
      - Hebrew RTL PDFs sometimes reorder bits of '(א) (15 נקודות)'.
      - This parser is tailored to your exam PDF format (questions + 'תשובה:').
    """
    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text("text") for page in doc)

    # Question headers like: (נקודות35) .1
    q_header_re = re.compile(r"\(נקודות\s*(\d+)\)\s*\.(\d+)")
    q_headers = [(m.start(), m.end(), int(m.group(2)), int(m.group(1))) for m in q_header_re.finditer(text)]

    # Part headers typically look like: ... נקודות( תהא20) ()ב
    part_re = re.compile(r"נקודות\(\s*.*?(\d+)\)\s*\(\)([א-ת])")

    # Answer marker
    ans_re = re.compile(r"תשובה\s*:|:תשובה")

    items: List[Dict[str, Any]] = []
    for idx, (qs, qe, qnum, _qpts) in enumerate(q_headers):
        q_end = q_headers[idx + 1][0] if idx + 1 < len(q_headers) else len(text)
        q_block = text[qe:q_end]

        # find all part lines
        part_headers = []
        for pm in part_re.finditer(q_block):
            line_start = q_block.rfind("\n", 0, pm.start()) + 1
            part_headers.append((line_start, pm.end(), pm.group(2), int(pm.group(1))))
        if not part_headers:
            # fallback: treat whole question as one item
            part_headers = [(0, 0, "", 0)]

        # sort + dedupe by start
        part_headers.sort(key=lambda x: x[0])
        ded = []
        for ph in part_headers:
            if not ded or ph[0] != ded[-1][0]:
                ded.append(ph)
        part_headers = ded

        for j, (pstart, _pend, part_letter, points_possible) in enumerate(part_headers):
            part_end = part_headers[j + 1][0] if j + 1 < len(part_headers) else len(q_block)
            part_text = q_block[pstart:part_end]

            am = ans_re.search(part_text)
            if am:
                qtext = part_text[:am.start()]
                sol = part_text[am.end():]
            else:
                qtext, sol = part_text, ""

            # remove the part marker clutter from the visible question text
            qtext = part_re.sub("", qtext)
            qtext = _clean_text(qtext)
            sol = _clean_text(sol)

            # keep
            items.append(
                {
                    "qnum": qnum,
                    "part": part_letter,
                    "points_possible": int(points_possible),
                    "question": qtext,
                    "solution": sol,
                }
            )

    # Keep only those that have real questions
    items = [it for it in items if it["question"]]
    return items

def extract_student_items(tex_path: Path) -> List[Dict[str, Any]]:
    """
    Extract student answers from the LaTeX file.
    Assumes the submission is organized as:
      \\subsection*{Question 1 ...}  -> 1א
      \\subsection*{Question 1 ...}  -> 1ב
      \\subsection*{Question 2 ...}  -> 2א
      \\subsection*{Question 2 ...}  -> 2ב
      \\subsection*{Question 3 ...}  -> 3א
      \\subsection*{Question 3 ...}  -> 3ב
    If your students use a different structure, adapt the mapping here.
    """
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"(\\subsection\*\{[^}]*\})", tex)
    subsections = []
    for i in range(1, len(blocks), 2):
        header = blocks[i]
        body = blocks[i + 1].strip()
        title = re.search(r"\\subsection\*\{([^}]*)\}", header).group(1)
        subsections.append((title, body))

    mapping = [(1, "א"), (1, "ב"), (2, "א"), (2, "ב"), (3, "א"), (3, "ב")]
    out = []
    for (qnum, part), (_title, body) in zip(mapping, subsections):
        out.append({"qnum": qnum, "part": part, "student_tex": body.strip()})
    return out

def compute_totals(graded_items: List[Dict[str, Any]]) -> Dict[str, int]:
    possible = sum(int(it.get("points_possible", 0)) for it in graded_items)
    awarded = sum(int(it.get("points_awarded", 0)) for it in graded_items)
    pct = 0 if possible == 0 else round(100 * awarded / possible)
    return {"possible": possible, "awarded": awarded, "percent": pct}

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
    s = "".join(out).replace("\r\n", "\n")
    return s.replace("\n", r"\par " + "\n")

def build_report_tex(model: str, system_prompt: str, user_prompt: str, grade: dict, totals: Dict[str, int]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    items = grade["items"]
    assumptions = escape_tex(grade.get("assumptions", ""))

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
    lines.append(r"{\Large \textbf{דו''ח בדיקה – בוחן}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(rf"\textbf{{תאריך:}} {escape_tex(now)}\par" "\n")
    lines.append(rf"\textbf{{מודל:}} {escape_tex(model)}\par" "\n")
    lines.append(rf"\textbf{{ציון סופי:}} {totals['awarded']}/{totals['possible']} ({totals['percent']}\%)\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\textbf{הנחות:}\par" "\n")
    lines.append(assumptions + r"\par" "\n")
    lines.append(r"\vspace{0.6cm}" "\n")

    # Prompts used (truncated)
    lines.append(r"\textbf{System prompt used:}\par" "\n")
    lines.append(r"\begin{quote}\ttfamily" "\n")
    lines.append(escape_tex(system_prompt) + "\n")
    lines.append(r"\end{quote}" "\n")
    lines.append(r"\textbf{User prompt used (first 1500 chars):}\par" "\n")
    lines.append(r"\begin{quote}\ttfamily" "\n")
    lines.append(escape_tex(user_prompt[:1500]) + "\n")
    lines.append(r"\end{quote}" "\n")
    lines.append(r"\newpage" "\n")

    # Breakdown table
    lines.append(r"{\Large \textbf{פירוט ציונים ומשוב}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\begin{longtable}{|p{0.22\textwidth}|p{0.14\textwidth}|p{0.56\textwidth}|}" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{נקודות} & \textbf{משוב לתלמיד/ה} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endfirsthead" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\textbf{סעיף} & \textbf{נקודות} & \textbf{משוב לתלמיד/ה} \\" "\n")
    lines.append(r"\hline" "\n")
    lines.append(r"\endhead" "\n")

    for it in items:
        title = escape_tex(it.get("title", ""))
        p_aw = int(it.get("points_awarded", 0))
        p_ps = int(it.get("points_possible", 0))
        question = escape_tex(it.get("question", ""))
        student_answer = escape_tex(it.get("student_answer", ""))
        feedback = escape_tex(it.get("feedback_he", ""))
        errors = it.get("errors", []) or []

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

        lines.append(f"{title} & {p_aw}/{p_ps} & {cell} \\\\" "\n")
        lines.append(r"\hline" "\n")

    lines.append(r"\end{longtable}" "\n")
    lines.append(r"\newpage" "\n")

    # Append original student PDF (compiled cleanly)
    lines.append(r"{\Large \textbf{הגשה מקורית (PDF כפי שהודפס מ-LaTeX)}}\par" "\n")
    lines.append(r"\vspace{0.3cm}" "\n")
    lines.append(r"\includepdf[pages=-]{012_clean.pdf}" "\n")

    lines.append(r"\end{document}" "\n")
    return "".join(lines)

def compile_report_tex():
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={BUILD_DIR}", str(GRADED_TEX)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={BUILD_DIR}", str(GRADED_TEX)])

# ---------- main ----------
def main():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_TEX.exists():
        raise FileNotFoundError(f"Missing {INPUT_TEX.resolve()}")
    if not REFERENCE_PDF.exists():
        raise FileNotFoundError(f"Missing {REFERENCE_PDF.resolve()}")

    # 1) Compile student submission using your existing compile_tex.py
    print("Step 1: Compiling student submission with compile_tex.py ...")
    run(["python", "compile_tex.py"])

    if not CLEAN_PDF.exists():
        raise RuntimeError("Expected build/012_clean.pdf was not created. Check compile_tex.py output.")

    # 2) Extract reference (questions + solutions) and student's answers
    print("Step 2: Extracting reference items from PDF ...")
    ref_items = extract_reference_items(REFERENCE_PDF)

    print("Step 3: Extracting student items from LaTeX ...")
    stu_items = extract_student_items(INPUT_TEX)

    # Align by (qnum, part). If you need a different mapping, change extract_student_items().
    ref_by_key = {(it["qnum"], it["part"]): it for it in ref_items}
    payload_items = []
    for s in stu_items:
        key = (s["qnum"], s["part"])
        r = ref_by_key.get(key)
        if not r:
            continue
        payload_items.append(
            {
                "title": f"שאלה {r['qnum']}({r['part']})",
                "qnum": r["qnum"],
                "part": r["part"],
                "points_possible": r["points_possible"],
                "question": r["question"],
                "official_solution": r["solution"],
                "student_answer_latex": s["student_tex"],
            }
        )

    payload_text = json.dumps(payload_items, ensure_ascii=False, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(payload=payload_text)

    # 4) Grade using ONE model
    print(f"Step 4: Grading with model: {MODEL} ...")
    grade = call_ollama_grade(MODEL, SYSTEM_PROMPT, user_prompt)

    # 5) Compute totals in Python (don’t trust model totals)
    totals = compute_totals(grade.get("items", []))

    # Save grading JSON for debugging
    (BUILD_DIR / "grade.json").write_text(json.dumps(grade, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) Build LaTeX report (feedback + embedded student PDF)
    print("Step 5: Building graded_report.tex ...")
    tex_report = build_report_tex(MODEL, SYSTEM_PROMPT, user_prompt, grade, totals)
    GRADED_TEX.write_text(tex_report, encoding="utf-8")

    # 7) Compile LaTeX report to final PDF
    print("Step 6: Compiling graded_report.pdf ...")
    compile_report_tex()

    if not GRADED_PDF.exists():
        raise RuntimeError("graded_report.pdf not created. Check LaTeX logs in build/.")

    print("\nDone.")
    print(f"- Clean student PDF: {CLEAN_PDF}")
    print(f"- Grading JSON      : {BUILD_DIR / 'grade.json'}")
    print(f"- Final graded PDF  : {GRADED_PDF}")

if __name__ == "__main__":
    main()
