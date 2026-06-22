from __future__ import annotations

"""Build machine-readable grading payloads.

Public API:
    build_payloads(...)

Everything else in this module is private.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from services.file_handling.part_normalize import normalize_part
from services.file_handling.pdf_cleanse import cleanse_test_pdf
from services.file_handling.reference_ranges import Key, find_reference_ranges
from services.file_handling.reference_tex import parse_reference_tex
from services.file_handling.student_tex import parse_student_tex_answers


@dataclass(frozen=True)
class PayloadItem:
    key: Key
    qid: str
    max_points: float
    reference_text: str
    student_latex: str
    payload_path: Path


_ANSWER_SPLIT_RE = re.compile(r"\bתשובה\s*:\s*", re.UNICODE)
_POINTS_AFTER_RE = re.compile(r"נקודות[^0-9]{0,12}(\d{1,3})", re.UNICODE)
_POINTS_BEFORE_RE = re.compile(r"(\d{1,3})\s*נקודות", re.UNICODE)
_MATH_BLOCK_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

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

_MATHY_RE = re.compile(
    r"(\\(frac|int|sum|lim|max|min|sup|inf|cdot|to|infty|Delta|delta|epsilon|lambda|pi|theta|alpha|beta|gamma)\b|"
    r"[A-Za-z]\w*\s*\([^)]*\)|"
    r"[A-Za-z]\w*_\{[^}]+\}|"
    r"[A-Za-z]\w*\^\{[^}]+\}|"
    r"[A-Za-z]\w*_[A-Za-z0-9]+|"
    r"[A-Za-z]\w*\^[A-Za-z0-9]+|"
    r"\b\d+\s*/\s*\d+\b|"
    r"[=<>]=?|\\leq|\\geq|\\neq)"
)

_DOC_WRAPPER_RX = re.compile(
    r"(?is)"
    r"(\\documentclass\b.*?\n)"
    r"|"
    r"(\\begin\{document\})"
    r"|"
    r"(\\end\{document\})"
)


def build_payloads(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 0.0,
) -> Tuple[Path, List[PayloadItem]]:
    """
    Public entry point.

    Supports:
      - reference PDF
      - reference TeX / TXT

    Returns:
      - manifest.json path
      - list of PayloadItem
    """
    suffix = reference_tex.suffix.lower()

    if suffix == ".pdf":
        return _build_payloads_from_reference_pdf(
            reference_pdf=reference_tex,
            student_tex=student_tex,
            out_dir=out_dir,
            default_max_points=default_max_points,
        )

    if suffix in {".tex", ".txt"}:
        return _build_payloads_from_reference_tex(
            reference_tex=reference_tex,
            student_tex=student_tex,
            out_dir=out_dir,
            default_max_points=default_max_points,
        )

    raise ValueError(
        f"Unsupported reference type: {reference_tex.name}. Expected .pdf or .tex/.txt"
    )


def _normalize_key(key: Key) -> Key:
    qnum, part = key
    return int(qnum), normalize_part(part)


def _qid_from_key(key: Key) -> str:
    qnum, part = _normalize_key(key)
    return f"Q{qnum}({part})" if part else f"Q{qnum}"


def _extract_reference_text(reference_pdf: Path, start_page_1b: int, end_page_1b: int) -> str:
    """Extract selectable text from a 1-based inclusive page range."""
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


def _split_question_and_solution(reference_block: str) -> tuple[str, str]:
    """
    Split a reference block into (question_text, solution_text).

    If no 'תשובה:' marker is found, return ("", reference_block) for compatibility.
    """
    txt = (reference_block or "").strip()
    if not txt:
        return "", ""

    match = _ANSWER_SPLIT_RE.search(txt)
    if not match:
        return "", txt

    question_text = txt[: match.start()].strip()
    solution_text = txt[match.end() :].strip()
    return question_text, solution_text


def _infer_points(text: str, default_max: float = 0.0) -> float:
    """Extract max points from a text block."""
    if not text:
        return default_max

    candidates: List[int] = []
    for rx in (_POINTS_AFTER_RE, _POINTS_BEFORE_RE):
        for match in rx.finditer(text):
            try:
                g_start = match.start(1)
                if g_start > 0 and text[g_start - 1] == ".":
                    continue
                candidates.append(int(match.group(1)))
            except Exception:
                pass

    if not candidates:
        return default_max

    return float(min(candidates))


def _infer_points_for_part(question_text: str, part: str, default_max: float = 0.0) -> float:
    """Prefer the points that belong to a specific part if detectable."""
    if not question_text or not part:
        return default_max

    idx = question_text.find(part)
    if idx == -1:
        return default_max

    window = question_text[max(0, idx - 50) : idx + 350]
    return _infer_points(window, default_max)


def _normalize_reference_to_latexish(text: str) -> str:
    """
    Best-effort cleanup for PDF-extracted math.

    Conservative:
      - replace common Unicode math symbols
      - wrap obvious math fragments in $...$
      - normalize whitespace
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for src, dst in _UNICODE_REPL.items():
        text = text.replace(src, dst)

    text = re.sub(r"\bdelta\s*\(", r"\\delta(", text, flags=re.IGNORECASE)
    text = re.sub(r"\bepsilon\b", r"\\epsilon", text, flags=re.IGNORECASE)
    text = re.sub(r"\blambda\s*\(", r"\\lambda(", text, flags=re.IGNORECASE)

    parts = _MATH_BLOCK_RE.split(text)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
        else:
            out.append(_MATHY_RE.sub(lambda m: f"${m.group(0)}$", part))

    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _infer_required_outcome(question_text: str, solution_text: str) -> str:
    """Heuristic short target string for the LLM."""
    q = (question_text or "").strip()
    s = (solution_text or "").strip()

    candidates = q.splitlines()[:12] + s.splitlines()[:8]
    for line in candidates:
        line = line.strip()
        if not line:
            continue
        if any(tok in line for tok in ["\\leq", "\\geq", "<=", ">=", "=", "\\int", "\\sum", "\\lim"]):
            return line[:300]

    for line in q.splitlines():
        line = line.strip()
        if line:
            return line[:300]

    return ""


def _strip_tex_document_wrappers(tex: str) -> str:
    """Remove TeX document wrapper commands from a fragment."""
    if not tex:
        return ""
    tex = tex.replace("\r\n", "\n").replace("\r", "\n")
    tex = _DOC_WRAPPER_RX.sub("", tex)
    tex = tex.replace(r"\end{document}", "").replace(r"\begin{document}", "")
    return tex.strip()


def _write_payload(
    *,
    payload_dir: Path,
    qid: str,
    payload: dict,
) -> Path:
    payload_path = payload_dir / f"{qid.replace('(', '_').replace(')', '')}.json"
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload_path


def _write_manifest(
    *,
    out_dir: Path,
    manifest: dict,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _build_payloads_from_reference_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 0.0,
) -> Tuple[Path, List[PayloadItem]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    ref_ranges_raw = find_reference_ranges(reference_clean_pdf, out_dir)
    ref_ranges: Dict[Key, Tuple[int, int]] = {
        _normalize_key(key): value for key, value in ref_ranges_raw.items()
    }
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed. "
            "Check out/debug_reference_pages.txt. "
            "If empty, the PDF may be scanned (no selectable text)."
        )

    student_answers_raw, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    student_answers: Dict[Key, str] = {
        _normalize_key(key): value for key, value in student_answers_raw.items()
    }
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX. "
            "Check out/debug_student_tex_parts.txt."
        )

    keys = sorted(set(ref_ranges.keys()) & set(student_answers.keys()))
    if not keys:
        raise RuntimeError(
            "No overlapping questions found between reference PDF ranges and parsed student TeX answers. "
            "(Common causes: question numbering mismatch, or missing part markers in the student template.)"
        )

    items: List[PayloadItem] = []

    for key in keys:
        start_p, end_p = ref_ranges[key]
        ref_text = _extract_reference_text(reference_clean_pdf, start_p, end_p)
        question_text, solution_text = _split_question_and_solution(ref_text)

        max_points = _infer_points_for_part(question_text, key[1], default_max_points)
        if not max_points:
            max_points = _infer_points(question_text, default_max_points)
        if not max_points:
            max_points = _infer_points(ref_text, default_max_points)

        student_latex = (student_answers.get(key) or "").strip()
        qid = _qid_from_key(key)

        question_latexish = _normalize_reference_to_latexish(question_text)
        solution_latexish = _normalize_reference_to_latexish(solution_text)
        required_outcome = _normalize_reference_to_latexish(
            _infer_required_outcome(question_text, solution_text)
        )

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
                "text": ref_text,
            },
            "student": {
                "source": "student.tex",
                "latex_raw": student_latex,
                "latex_clean": "",
            },
            "ai_input": {
                "question_latex": question_latexish,
                "reference_solution_latex": solution_latexish,
                "student_answer_latex": student_latex,
                "required_outcome": required_outcome,
                "max_points": float(max_points),
            },
        }

        payload_path = _write_payload(
            payload_dir=payload_dir,
            qid=qid,
            payload=payload,
        )

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
        "reference_pdf": reference_pdf.name,
        "student_tex": student_tex.name,
        "count": len(items),
        "items": [
            {"qid": item.qid, "payload_file": item.payload_path.name}
            for item in items
        ],
    }

    manifest_path = _write_manifest(out_dir=out_dir, manifest=manifest)
    return manifest_path, items


def _build_payloads_from_reference_tex(
    *,
    reference_tex: Path,
    student_tex: Path,
    out_dir: Path,
    default_max_points: float = 100.0,
) -> Tuple[Path, List[PayloadItem]]:
    """
    Create payloads from a reference .tex/.txt + student .tex.

    This avoids PDF extraction entirely. Reference content is already LaTeX,
    so it is passed through without escaping.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    reference_tex_text = reference_tex.read_text(encoding="utf-8", errors="replace")
    ref_parts_raw = parse_reference_tex(reference_tex_text)
    ref_parts = {_normalize_key(key): value for key, value in ref_parts_raw.items()}
    if not ref_parts:
        raise RuntimeError(
            "Could not parse any questions/parts from reference .tex. "
            "Expected headings like \\section*{Question N} and \\subsection*{(a)}."
        )

    student_answers_raw, _student_ranges = parse_student_tex_answers(student_tex, out_dir)
    student_answers = {_normalize_key(key): value for key, value in student_answers_raw.items()}
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX. "
            "Check out/debug_student_tex_parts.txt."
        )

    keys = sorted(set(ref_parts.keys()) | set(student_answers.keys()))
    items: List[PayloadItem] = []

    for key in keys:
        qid = _qid_from_key(key)

        ref = ref_parts.get(key)
        question_tex = _strip_tex_document_wrappers((ref.title if ref else "").strip())
        solution_tex = _strip_tex_document_wrappers((ref.latex_body if ref else "").strip())
        reference_block = "\n\n".join(
            part for part in [question_tex, solution_tex] if part
        ).strip()

        if r"\end{document}" in reference_block or r"\begin{document}" in reference_block:
            raise RuntimeError(
                f"Reference TeX fragment for {qid} still contains document wrapper commands."
            )

        student_latex = (student_answers.get(key) or "").strip()
        max_points = float(default_max_points or 0.0)

        required_outcome = _normalize_reference_to_latexish(
            _infer_required_outcome(question_tex, solution_tex)
        )

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
                "question_latex": question_tex,
                "reference_solution_latex": solution_tex,
                "student_answer_latex": student_latex,
                "required_outcome": required_outcome,
                "max_points": float(max_points),
            },
        }

        payload_path = _write_payload(
            payload_dir=payload_dir,
            qid=qid,
            payload=payload,
        )

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
        "reference_tex": reference_tex.name,
        "student_tex": student_tex.name,
        "count": len(items),
        "items": [
            {"qid": item.qid, "payload_file": item.payload_path.name}
            for item in items
        ],
    }

    manifest_path = _write_manifest(out_dir=out_dir, manifest=manifest)
    return manifest_path, items