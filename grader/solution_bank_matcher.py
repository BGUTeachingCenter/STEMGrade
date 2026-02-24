from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from grader.reference_tex import parse_reference_tex
from grader.student_tex import parse_student_tex_answers
from grader.ai_grading.ollama_client import OllamaClient


@dataclass(frozen=True)
class CandidateRef:
    ref_id: str
    path: Path
    summary: str
    qnums: Tuple[int, ...]  # for cheap matching


def _collect_reference_tex_files(bank_dir: Path) -> List[Path]:
    if not bank_dir.exists():
        return []
    # You can restrict patterns if you want (e.g. only */reference.tex)
    return sorted([p for p in bank_dir.rglob("*.tex") if p.is_file()])


def _summarize_reference_tex(path: Path, max_items: int = 18) -> CandidateRef:
    parts = parse_reference_tex(path)  # Dict[(qnum, part)] -> RefPart
    qnums = sorted({q for (q, _part) in parts.keys()})
    # Build a compact summary: question numbers + a few part titles
    items = []
    for (q, part), rp in list(parts.items())[:max_items]:
        title = (rp.title or "").strip()
        title = re.sub(r"\s+", " ", title)
        if title:
            items.append(f"Q{q}{part}: {title[:80]}")
        else:
            items.append(f"Q{q}{part}")
    summary = f"file={path.name}; qnums={qnums}; parts_preview={items}"
    # stable id from relative path
    ref_id = str(path).replace("\\", "/")
    return CandidateRef(ref_id=ref_id, path=path, summary=summary, qnums=tuple(qnums))


def _heuristic_match_by_qnums(student_tex: Path, candidates: List[CandidateRef]) -> Optional[CandidateRef]:
    """
    If student TeX has parseable question numbers, pick reference with best overlap.
    """
    try:
        student_answers, _ranges = parse_student_tex_answers(student_tex, student_tex.parent)
    except Exception:
        return None
    if not student_answers:
        return None

    student_qnums = sorted({q for (q, _part) in student_answers.keys()})
    if not student_qnums:
        return None

    best = None
    best_score = -1
    for c in candidates:
        overlap = len(set(student_qnums) & set(c.qnums))
        # penalize extra questions a bit, prefer tighter match
        extra = max(0, len(set(c.qnums) - set(student_qnums)))
        score = overlap * 10 - extra
        if score > best_score:
            best_score = score
            best = c

    # require at least some overlap
    if best and best_score >= 10:
        return best
    return None


def _llm_match(student_tex: Path, candidates: List[CandidateRef], *, top_k: int = 12) -> CandidateRef:
    """
    Ask Ollama to pick the best reference for this student submission.
    Uses only candidate summaries (no full solutions) to keep prompt short.
    """
    text = student_tex.read_text(encoding="utf-8", errors="replace")
    # keep it short
    student_snip = text[:6000]

    # limit candidates in prompt
    short_list = candidates[:top_k]
    options = [{"ref_id": c.ref_id, "summary": c.summary} for c in short_list]

    schema = {
        "type": "object",
        "properties": {
            "ref_id": {"type": "string"},
            "confidence": {"type": "number"},
            "why": {"type": "string"},
        },
        "required": ["ref_id", "confidence", "why"],
        "additionalProperties": False,
    }

    system = (
        "You select which teacher reference.tex matches a student's submitted answers.tex.\n"
        "Return JSON only. Pick the best ref_id from the provided options.\n"
        "If unsure, still pick the closest match and lower confidence."
    )
    user = (
        "Student submission (truncated):\n"
        f"{student_snip}\n\n"
        "Reference candidates:\n"
        f"{json.dumps(options, ensure_ascii=False, indent=2)}\n\n"
        "Choose the best ref_id."
    )

    client = OllamaClient()
    out = client.chat_json(system=system, user=user, schema=schema, temperature=0.1, timeout_s=600)

    chosen = out.get("ref_id", "")
    for c in short_list:
        if c.ref_id == chosen:
            return c

    # fallback if model returned something unexpected
    return short_list[0]


def pick_reference_from_bank(
    *,
    bank_dir: Path,
    student_tex: Path,
    prefer_heuristic: bool = True,
    llm_top_k: int = 12,
) -> Path:
    """
    Returns a path to the chosen reference.tex in the bank.
    """
    ref_files = _collect_reference_tex_files(bank_dir)
    if not ref_files:
        raise RuntimeError(f"Solution bank is empty or missing: {bank_dir}")

    candidates = [_summarize_reference_tex(p) for p in ref_files]

    if prefer_heuristic:
        h = _heuristic_match_by_qnums(student_tex, candidates)
        if h:
            return h.path

    chosen = _llm_match(student_tex, candidates, top_k=llm_top_k)
    return chosen.path