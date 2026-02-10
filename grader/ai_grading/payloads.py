from __future__ import annotations

"""Build machine-readable grading payloads.

We separate *data preparation* from *AI grading*.

Payloads are JSON files (one per question/part) plus a manifest.json that
enumerates them. This makes the pipeline reproducible and debuggable:

  reference.pdf + student.tex -> payloads/*.json -> grades.json -> graded PDF

Design goals:
  - Hebrew stays as Unicode text.
  - Math stays as LaTeX when possible (student answer is raw LaTeX).
  - Each question/part is a standalone unit with a stable id.
  - Provide an AI-focused compact view ("ai_input") to reduce LLM confusion.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from grader.pdf_cleanse import cleanse_test_pdf
from grader.reference_ranges import Key, find_reference_ranges
from grader.reference_tex import parse_reference_tex
from grader.student_tex import parse_student_tex_answers


_ANSWER_SPLIT_RE = re.compile(r"\bתשובה\s*:\s*", re.UNICODE)
_POINTS_AFTER_RE = re.compile(r"נקודות[^0-9]{0,12}(\d{1,3})", re.UNICODE)
_POINTS_BEFORE_RE = re.compile(r"(\d{1,3})\s*נקודות", re.UNICODE)

# If the reference text already contains LaTeX math blocks, we preserve them.
_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

# Unicode -> LaTeX-ish substitutions (conservative)
_UNICODE_REPL = {
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "→": r"\to",
    "∞": r"\infty",
    "×": r"\cdot",
    "·": r"\cdot",
    "∈": r"\in",
    "∉": r"\notin",
    "∪": r"\cup",
    "∩": r"\cap",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∑": r"\sum",
    "∫": r"\int",
    "∂": r"\partial",
    "∆": r"\Delta",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "λ": r"\lambda",
    "π": r"\pi",
    "θ": r"\theta",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
}

# Heuristic "mathy fragment" detector (conservative: wraps only obvious tokens)
_MATHY_RE = re.compile(
    r"(\\(frac|int|sum|lim|max|min|sup|inf|cdot|to|infty|Delta|delta|epsilon|lambda|pi|theta|alpha|beta|gamma)\b|"
    r"[A-Za-z]\w*\s*\([^)]*\)|"             # f(x), S(...), lim(...)
    r"[A-Za-z]\w*_\{[^}]+\}|"               # x_{i+1}
    r"[A-Za-z]\w*\^\{[^}]+\}|"              # x^{2}
    r"[A-Za-z]\w*_[A-Za-z0-9]+|"            # x_i
    r"[A-Za-z]\w*\^[A-Za-z0-9]+|"           # x^2
    r"\b\d+\s*/\s*\d+\b|"                   # 2/3
    r"[=<>]=?|\\leq|\\geq|\\neq)"           # comparisons
)


@dataclass(frozen=True)
class PayloadItem:
    key: Key
    qid: str
    max_points: float
    reference_text: str
    student_latex: str
    payload_path: Path


def qid_from_key(key: Key) -> str:
    qnum, part = key
    if part:
        return f"Q{qnum}({part})"
    return f"Q{qnum}"


def extract_reference_text(reference_pdf: Path, start_page_1b: int, end_page_1b: int) -> str:
    """Extract selectable text from a page range (1-based inclusive)."""
    doc = fitz.open(str(reference_pdf))
    try:
        parts: List[str] = []
        start0 = max(0, start_page_1b - 1)
        end0 = min(doc.page_count - 1, end_page_1b - 1)
        for i in range(start0, end0 + 1):
            txt = (doc[i].get_text("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if txt:
                parts.append(txt)
        return "\n\n".join(parts).strip()
    finally:
        doc.close()


def split_question_and_solution(reference_block: str) -> tuple[str, str]:
    """Split a reference block into (question_text, solution_text).

    The reference PDFs we work with usually contain the question statement
    followed by a Hebrew marker like "תשובה:" (official solution).

    If the marker is not found, we return ("", reference_block) to preserve
    backward compatibility.
    """
    txt = (reference_block or "").strip()
    if not txt:
        return "", ""

    m = _ANSWER_SPLIT_RE.search(txt)
    if not m:
        return "", txt

    q = txt[: m.start()].strip()
    sol = txt[m.end() :].strip()
    return q, sol


def infer_points(text: str, default_max: float = 0.0) -> float:
    """Extract max points from a reference text block."""
    if not text:
        return default_max
    candidates: List[int] = []
    for rx in (_POINTS_AFTER_RE, _POINTS_BEFORE_RE):
        for m in rx.finditer(text):
            try:
                # Avoid false positives like the question number in strings
                # such as "(נקודות35) .1" (the ".1" is not points).
                g_start = m.start(1)
                if g_start > 0 and text[g_start - 1] == ".":
                    continue
                candidates.append(int(m.group(1)))
            except Exception:
                pass
    if not candidates:
        return default_max
    # In many PDFs the total question points appear before the part points.
    # Returning the *smallest* candidate tends to pick the part points (e.g. 15)
    # over the total (e.g. 35). For questions that have no parts, this still
    # returns the correct total.
    return float(min(candidates))


def infer_points_for_part(question_text: str, part: str, default_max: float = 0.0) -> float:
    """Prefer the points that belong to a specific part (א/ב) if detectable.

    Some RTL PDFs put the total question points before the part points.
    We try to locate the part marker and then search for "נקודות" nearby.
    """
    if not question_text or not part:
        return default_max

    idx = question_text.find(part)
    if idx == -1:
        return default_max

    window = question_text[max(0, idx - 50) : idx + 350]
    return infer_points(window, default_max)


def _normalize_reference_to_latexish(s: str) -> str:
    """Best-effort: make PDF-extracted math more readable for LLMs.

    This is intentionally conservative: it does NOT invent structure.
    It only:
      - converts common unicode math symbols to LaTeX-ish tokens
      - wraps obvious math fragments in $...$
      - normalizes whitespace
    """
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Replace unicode math symbols everywhere
    for k, v in _UNICODE_REPL.items():
        s = s.replace(k, v)

    # Fix a few common plain-text math words
    s = re.sub(r"\bdelta\s*\(", r"\\delta(", s, flags=re.IGNORECASE)
    s = re.sub(r"\bepsilon\b", r"\\epsilon", s, flags=re.IGNORECASE)
    s = re.sub(r"\blambda\s*\(", r"\\lambda(", s, flags=re.IGNORECASE)

    # Wrap mathy fragments in $...$ but DO NOT touch existing $...$
    parts = _MATH_BLOCK_RE.split(s)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
        else:
            out.append(_MATHY_RE.sub(lambda m: f"${m.group(0)}$", part))
    cleaned = "".join(out)

    # Light whitespace normalization (avoid one-token-per-line look)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _infer_required_outcome(question_text: str, solution_text: str) -> str:
    """Heuristic: produce a short 'required outcome' string to anchor the LLM.

    Prefer a line containing an equation/inequality or integral/sum/limit.
    """
    q = (question_text or "").strip()
    s = (solution_text or "").strip()

    candidates = (q.splitlines()[:12]) + (s.splitlines()[:8])
    for line in candidates:
        line2 = line.strip()
        if not line2:
            continue
        if any(tok in line2 for tok in ["\\leq", "\\geq", "<=", ">=", "=", "\\int", "\\sum", "\\lim"]):
            return line2[:300]

    for line in q.splitlines():
        if line.strip():
            return line.strip()[:300]

    return ""


def build_payloads(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 0.0,
) -> Tuple[Path, List[PayloadItem]]:
    """Create per-question JSON payloads + a manifest.

    Returns:
      - manifest.json path
      - list of created PayloadItem
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    # 1) Clean reference and detect per-question ranges
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    ref_ranges = find_reference_ranges(reference_clean_pdf, out_dir)
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed. "
            "Check out/debug_reference_pages.txt. "
            "If empty, the PDF may be scanned (no selectable text)."
        )

    # 2) Parse student's TeX into snippets per question/part
    student_answers, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX. "
            "Check out/debug_student_tex_parts.txt."
        )

    # 3) Create payloads only for overlapping keys
    keys = sorted(set(ref_ranges.keys()) & set(student_answers.keys()))
    if not keys:
        raise RuntimeError(
            "No overlapping questions found between reference PDF ranges and parsed student TeX answers. "
            "(Common causes: question numbering mismatch, or missing part markers in the student template.)"
        )

    items: List[PayloadItem] = []

    for key in keys:
        start_p, end_p = ref_ranges[key]
        ref_text = extract_reference_text(reference_clean_pdf, start_p, end_p)
        question_text, solution_text = split_question_and_solution(ref_text)

        # Prefer points that belong to the specific part (א/ב), because many PDFs show total question points first.
        max_points = infer_points_for_part(question_text, key[1], default_max_points)
        if not max_points:
            max_points = infer_points(question_text, default_max_points)
        if not max_points:
            max_points = infer_points(ref_text, default_max_points)

        student_latex = (student_answers.get(key) or "").strip()
        qid = qid_from_key(key)

        # NEW: compact AI-facing fields (reduce LLM confusion)
        question_latexish = _normalize_reference_to_latexish(question_text)
        solution_latexish = _normalize_reference_to_latexish(solution_text)
        required_outcome = _normalize_reference_to_latexish(_infer_required_outcome(question_text, solution_text))

        payload = {
            "question_id": qid,
            "qid": qid,
            "key": {"qnum": key[0], "part": key[1]},
            "max_points": float(max_points),
            "rubric": {"score_max": float(max_points), "key_points": []},
            "reference": {
                "source": "reference.pdf",
                "page_range_1_based_inclusive": [start_p, end_p],
                "question_text": question_text,
                "solution_text": solution_text,
                # Backward-compatible combined text (debug)
                "text": ref_text,
            },
            "student": {
                "source": "student.tex",
                "latex_raw": student_latex,
                "latex_clean": "",
            },
            # AI-focused compact view
            "ai_input": {
                "question_latex": question_latexish,
                "reference_solution_latex": solution_latexish,
                "student_answer_latex": student_latex,
                "required_outcome": required_outcome,
                "max_points": float(max_points),
            },
        }

        payload_path = payload_dir / f"{qid.replace('(', '_').replace(')', '')}.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        items.append(
            PayloadItem(
                key=key,
                qid=qid,
                max_points=float(max_points),
                reference_text=ref_text,
                student_latex=student_latex,
                payload_path=payload_path,
            )
        )

    manifest = {
        "version": 2,
        "reference_pdf": str(reference_pdf.name),
        "student_tex": str(student_tex.name),
        "count": len(items),
        "items": [
            {
                "qid": it.qid,
                "payload_file": it.payload_path.name,
            }
            for it in items
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest_path, items


def build_payloads_from_reference_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 0.0,
) -> Tuple[Path, List[PayloadItem]]:
    """Create payloads from a reference .tex (questions/solutions) + student .tex.

    This avoids PDF text extraction entirely. The reference content is assumed to
    be *LaTeX already*, so we feed it to the LLM without escaping.

    Notes:
      - We still use the student's raw LaTeX per part.
      - If the reference .tex doesn't contain explicit question text, the
        question_latex field may be empty; the solution_latex remains usable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    reference_tex_text = Path(reference_tex).read_text(encoding="utf-8", errors="replace")
    ref_parts = parse_reference_tex(reference_tex_text)

    if not ref_parts:
        raise RuntimeError(
            "Could not parse any questions/parts from reference .tex. "
            "Expected headings like \\section*{Question N} and \\subsection*{(a)}."
        )

    student_answers, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX. "
            "Check out/debug_student_tex_parts.txt."
        )

    # Union keys: we want payloads even if either side is missing (helps debugging).
    keys = sorted(set(ref_parts.keys()) | set(student_answers.keys()))
    items: List[PayloadItem] = []

    for key in keys:
        qid = qid_from_key(key)

        ref = ref_parts.get(key)
        question_tex = (ref.title if ref else "").strip()
        solution_tex = (ref.latex_body if ref else "").strip()
        reference_block = (question_tex + "\n\n" + solution_tex).strip() if (question_tex or solution_tex) else ""

        student_latex = (student_answers.get(key) or "").strip()

        # Points: reference .tex may not include points. Keep default_max_points.
        max_points = float(default_max_points or 0.0)

        # For consistency with the PDF path, we still provide the AI-facing fields.
        required_outcome = _normalize_reference_to_latexish(_infer_required_outcome(question_tex, solution_tex))

        payload = {
            "question_id": qid,
            "qid": qid,
            "key": {"qnum": key[0], "part": key[1]},
            "max_points": float(max_points),
            "rubric": {"score_max": float(max_points), "key_points": []},
            "reference": {
                "source": "reference.tex",
                "question_text": question_tex,
                "solution_text": solution_tex,
                "text": reference_block,
            },
            "student": {
                "source": "student.tex",
                "latex_raw": student_latex,
                "latex_clean": "",
            },
            "ai_input": {
                # These are already LaTeX; do NOT escape.
                "question_latex": question_tex,
                "reference_solution_latex": solution_tex,
                "student_answer_latex": student_latex,
                "required_outcome": required_outcome,
                "max_points": float(max_points),
            },
        }

        payload_path = payload_dir / f"{qid.replace('(', '_').replace(')', '')}.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        items.append(
            PayloadItem(
                key=key,
                qid=qid,
                max_points=float(max_points),
                reference_text=reference_block,
                student_latex=student_latex,
                payload_path=payload_path,
            )
        )

    manifest = {
        "version": 2,
        "reference_tex": str(reference_tex.name),
        "student_tex": str(student_tex.name),
        "count": len(items),
        "items": [{"qid": it.qid, "payload_file": it.payload_path.name} for it in items],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, items


def build_reference_snippets_from_payloads(payloads) -> dict[tuple[int,str], str]:
    ref_snips = {}
    for p in payloads:
        q = (p.reference.get("question_text") or "").strip()
        sol = (p.reference.get("solution_text") or "").strip()
        block = "\n\n".join([x for x in [q, sol] if x]).strip()
        ref_snips[p.key] = block
    return ref_snips

