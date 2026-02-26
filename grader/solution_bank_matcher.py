# solution_bank_matcher.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.storage import list_exam_summaries
from grader.student_tex import parse_student_tex_answers
from grader.ai_grading.ollama_client import OllamaClient
from grader.ai_grading.gpt_client import GptClient


@dataclass(frozen=True)
class CandidateRef:
    exam_id: str
    path: Path
    summary: str
    qnums: Tuple[int, ...]


def _collect_candidates_from_summaries(bank_dir: Path) -> List[CandidateRef]:
    """
    Build candidates from precomputed reference_summary.json.
    Fast: no TeX parsing at match time.
    """
    out: List[CandidateRef] = []

    for s in list_exam_summaries():
        exam_id = (s.get("exam_id") or "").strip()
        if not exam_id:
            continue

        qnums = tuple(int(x) for x in (s.get("qnums") or []) if str(x).isdigit())

        # By convention, reference lives here:
        # solution_bank/<exam_id>/uploads/reference_current.tex
        ref_path = bank_dir / exam_id / "uploads" / "reference_current.tex"
        if not ref_path.exists():
            continue

        parts_preview = s.get("parts_preview") or []
        summary = f"exam_id={exam_id}; qnums={list(qnums)}; parts_preview={parts_preview[:12]}"
        out.append(CandidateRef(exam_id=exam_id, path=ref_path, summary=summary, qnums=qnums))

    return out


def _heuristic_match_by_qnums(student_tex: Path, candidates: List[CandidateRef]) -> Optional[CandidateRef]:
    """
    If student TeX has parseable question numbers, pick reference with best overlap.
    Returns None if there is no confident match.
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

    best: CandidateRef | None = None
    best_score = -10**9
    sset = set(student_qnums)

    for c in candidates:
        cset = set(c.qnums)
        overlap = len(sset & cset)
        extra = len(cset - sset)
        missing = len(sset - cset)

        # tighter is better; missing hurts more than extra
        score = overlap * 10 - extra * 2 - missing * 4

        if score > best_score:
            best_score = score
            best = c

    # Require real overlap (>=1 question), and decent score
    if best and best_score >= 10:
        return best
    return None


def _llm_match(student_tex: Path, candidates: List[CandidateRef], *, top_k: int = 8) -> CandidateRef:
    """
    Ask Ollama to pick best match using only summaries.

    NO FALLBACK:
      - if model returns an exam_id not in options -> raise
      - if options empty -> raise
    """
    if not candidates:
        raise RuntimeError("No candidates available for LLM match.")

    text = student_tex.read_text(encoding="utf-8", errors="replace")
    student_snip = text[:2500]  # keep short

    short_list = candidates[:top_k]
    options = [{"exam_id": c.exam_id, "summary": c.summary[:700]} for c in short_list]

    schema = {
        "type": "object",
        "properties": {
            "exam_id": {"type": "string"},
            "confidence": {"type": "number"},
            "why": {"type": "string"},
        },
        "required": ["exam_id", "confidence", "why"],
        "additionalProperties": False,
    }

    system = (
        "You select which teacher exam_id matches a student's submitted answers.tex.\n"
        "Return JSON only. Pick the best exam_id from the provided options.\n"
        "If unsure, still pick the closest match and lower confidence."
    )
    user = (
        "Student submission (truncated):\n"
        f"{student_snip}\n\n"
        "Exam candidates:\n"
        f"{json.dumps(options, ensure_ascii=False, indent=2)}\n\n"
        "Choose the best exam_id."
    )

    client = OllamaClient()
    out = client.chat_json(system=system, user=user, schema=schema, temperature=0.1, timeout_s=90)

    chosen = (out.get("exam_id") or "").strip()
    if not chosen:
        raise RuntimeError("LLM did not return an exam_id.")

    for c in short_list:
        if c.exam_id == chosen:
            return c

    raise RuntimeError(
        f"LLM returned exam_id='{chosen}' which is not in the candidate list. "
        f"Candidates: {[c.exam_id for c in short_list]}"
    )


def pick_reference_from_bank(
    *,
    bank_dir: Path,
    student_tex: Path,
    prefer_heuristic: bool = True,
    llm_top_k: int = 8,
) -> Path:
    """
    Returns path to the chosen reference_current.tex using saved summaries.

    NO FALLBACK:
      - if heuristic is enabled and no match -> raise (unless you explicitly want LLM to run)
      - if LLM is run and doesn't pick a valid candidate -> raise
    """
    candidates = _collect_candidates_from_summaries(bank_dir)
    if not candidates:
        raise RuntimeError(
            "No reference summaries found in bank. "
            "Upload a teacher reference first (reference_current.tex) so reference_summary.json is created. "
            f"bank_dir={bank_dir}"
        )

    # Heuristic first
    if prefer_heuristic:
        h = _heuristic_match_by_qnums(student_tex, candidates)
        if h:
            return h.path

        # 🚫 No fallback: if heuristic is on and fails, stop here
        raise RuntimeError(
            "No confident match found by heuristic matching (qnums overlap). "
            "Student answers did not match any exam in the bank."
        )

    # If you turn off heuristic, we try LLM directly
    chosen = _llm_match(student_tex, candidates, top_k=llm_top_k)
    return chosen.path