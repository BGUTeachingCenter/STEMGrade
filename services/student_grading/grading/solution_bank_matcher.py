# solution_bank_matcher.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.storage import list_exam_summaries, write_reference_summary
from common.tex.student_tex import parse_student_tex_answers
from common.exam_summary import build_student_summary_from_answer_bundle, _fallback_student_qnums
from core.ai_clients.ollama_client import OllamaClient


@dataclass(frozen=True)
class CandidateRef:
    exam_id: str
    path: Path
    summary: str
    qnums: Tuple[int, ...]
    source: str = "unknown"
    part_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceMatch:
    exam_id: str
    path: Path
    source: str
    method: str
    reason: str
    student_qnums: Tuple[int, ...]
    reference_qnums: Tuple[int, ...]
    score: int | None = None


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

        # Preferred grading reference:
        #   solution_bank/<exam_id>/uploads/full_solution_bundle.json
        # Legacy fallback:
        #   solution_bank/<exam_id>/uploads/reference_current.tex
        uploads = bank_dir / exam_id / "uploads"
        json_path = uploads / "full_solution_bundle.json"
        tex_path = uploads / "reference_current.tex"

        if json_path.exists():
            # Existing banks may have an older TeX-derived reference_summary.json.
            # Refresh it in-place so matching compares the student summary to the
            # same JSON bundle that grading will use.
            if s.get("source") != "full_solution_bundle.json":
                try:
                    summary_path = write_reference_summary(exam_id)
                    refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(refreshed, dict):
                        s = refreshed
                except Exception:
                    pass
            ref_path = json_path
            source = "json"
        elif tex_path.exists():
            ref_path = tex_path
            source = "tex_fallback"
        else:
            continue

        parts_preview = s.get("parts_preview") or []
        part_keys = tuple(str(x) for x in (s.get("part_keys") or []) if str(x).strip())
        summary = (
            f"exam_id={exam_id}; reference_source={source}; "
            f"qnums={list(qnums)}; parts_preview={parts_preview[:12]}"
        )
        out.append(
            CandidateRef(
                exam_id=exam_id,
                path=ref_path,
                summary=summary,
                qnums=qnums,
                source=source,
                part_keys=part_keys,
            )
        )

    return out


def _student_qnums_for_matching(student_tex: Path) -> list[int]:
    """Extract question numbers for reference matching.

    Primary path uses student_answer_bundle.json when available. Legacy TeX
    parsing remains for manually uploaded .tex answers. Fallback regex is used
    only for selecting the exam, not for grading alignment.
    """
    if student_tex.suffix.lower() == ".json":
        try:
            summary = build_student_summary_from_answer_bundle(student_tex)
            qnums = [int(x) for x in (summary.get("qnums") or [])]
            if qnums:
                return sorted(set(qnums))
        except Exception:
            pass

    try:
        student_answers, _ranges = parse_student_tex_answers(student_tex, student_tex.parent)
        if student_answers:
            qnums = sorted({q for (q, _part) in student_answers.keys()})
            if qnums:
                return qnums
    except Exception:
        pass

    return _fallback_student_qnums(student_tex)


def _heuristic_match_detail_by_qnums(
    student_tex: Path,
    candidates: List[CandidateRef],
) -> tuple[CandidateRef, int, list[int]] | None:
    """
    If student TeX has parseable question numbers, pick reference with best overlap.
    Returns (candidate, score, student_qnums) or None if there is no confident match.
    """
    student_qnums = _student_qnums_for_matching(student_tex)
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
        return best, best_score, student_qnums
    return None


def _heuristic_match_by_qnums(student_tex: Path, candidates: List[CandidateRef]) -> Optional[CandidateRef]:
    detail = _heuristic_match_detail_by_qnums(student_tex, candidates)
    return detail[0] if detail else None


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


# def pick_reference_from_bank(
#     *,
#     bank_dir: Path,
#     student_tex: Path,
#     prefer_heuristic: bool = True,
#     llm_top_k: int = 8,
# ) -> Path:
#     """
#     Returns path to the chosen reference_current.tex using saved summaries.
#
#     NO FALLBACK:
#       - if heuristic is enabled and no match -> raise (unless you explicitly want LLM to run)
#       - if LLM is run and doesn't pick a valid candidate -> raise
#     """
#     candidates = _collect_candidates_from_summaries(bank_dir)
#     if not candidates:
#         raise RuntimeError(
#             "No reference summaries found in bank. "
#             "Upload a teacher reference first (reference_current.tex) so reference_summary.json is created. "
#             f"bank_dir={bank_dir}"
#         )
#
#     # Heuristic first
#     if prefer_heuristic:
#         h = _heuristic_match_by_qnums(student_tex, candidates)
#         if h:
#             return h.path
#
#         # 🚫 No fallback: if heuristic is on and fails, stop here
#         raise RuntimeError(
#             "No confident match found by heuristic matching (qnums overlap). "
#             "Student answers did not match any exam in the bank."
#         )
#
#     # If you turn off heuristic, we try LLM directly
#     chosen = _llm_match(student_tex, candidates, top_k=llm_top_k)
#     return chosen.path

def pick_reference_with_match_info(
    *,
    bank_dir: Path,
    student_tex: Path,
    prefer_heuristic: bool = True,
    llm_top_k: int = 8,
) -> ReferenceMatch:
    """
    Returns structured match metadata for debug traces and summary_compare.json.

    The selected path prefers full_solution_bundle.json when it exists.
    reference_current.tex is used only as a legacy fallback.
    """
    candidates = _collect_candidates_from_summaries(bank_dir)
    if not candidates:
        raise RuntimeError(
            "No reference summaries found in bank. "
            "Upload a teacher reference first so reference_summary.json is created. "
            f"bank_dir={bank_dir}"
        )

    student_qnums = tuple(_student_qnums_for_matching(student_tex))

    if prefer_heuristic:
        detail = _heuristic_match_detail_by_qnums(student_tex, candidates)
        if detail:
            candidate, score, detail_student_qnums = detail
            return ReferenceMatch(
                exam_id=candidate.exam_id,
                path=candidate.path,
                source=candidate.source,
                method="heuristic_qnums",
                reason="question_number_overlap",
                student_qnums=tuple(detail_student_qnums),
                reference_qnums=candidate.qnums,
                score=score,
            )

        # Controlled fallback for the common development/teacher-test case:
        # if the bank has exactly one exam, use it instead of failing before
        # grading. This is safer than guessing among multiple exams and lets us
        # continue testing OCR -> JSON grading end-to-end.
        if len(candidates) == 1:
            only = candidates[0]
            return ReferenceMatch(
                exam_id=only.exam_id,
                path=only.path,
                source=only.source,
                method="single_candidate_fallback",
                reason="only_one_exam_in_bank",
                student_qnums=student_qnums,
                reference_qnums=only.qnums,
                score=None,
            )

        # If there are multiple exams, ask the lightweight matcher model.
        # If that also fails, surface a useful error.
        try:
            chosen = _llm_match(student_tex, candidates, top_k=llm_top_k)
            return ReferenceMatch(
                exam_id=chosen.exam_id,
                path=chosen.path,
                source=chosen.source,
                method="llm_fallback",
                reason="heuristic_qnums_failed",
                student_qnums=student_qnums,
                reference_qnums=chosen.qnums,
                score=None,
            )
        except Exception as e:
            raise RuntimeError(
                "No confident match found by heuristic matching (qnums overlap), "
                "and LLM fallback could not choose an exam. "
                "Student answers may be missing clear markers like "
                "\\subsection*{Question 1} or \\subsection*{שאלה 1}. "
                f"Fallback error: {e}"
            ) from e

    chosen = _llm_match(student_tex, candidates, top_k=llm_top_k)
    return ReferenceMatch(
        exam_id=chosen.exam_id,
        path=chosen.path,
        source=chosen.source,
        method="llm",
        reason="heuristic_disabled",
        student_qnums=student_qnums,
        reference_qnums=chosen.qnums,
        score=None,
    )


def pick_reference_with_exam_id(
    *,
    bank_dir: Path,
    student_tex: Path,
    prefer_heuristic: bool = True,
    llm_top_k: int = 8,
) -> tuple[str, Path]:
    """
    Backwards-compatible: returns (exam_id, reference_path).
    """
    match = pick_reference_with_match_info(
        bank_dir=bank_dir,
        student_tex=student_tex,
        prefer_heuristic=prefer_heuristic,
        llm_top_k=llm_top_k,
    )
    return match.exam_id, match.path

def pick_reference_from_bank(
    *,
    bank_dir: Path,
    student_tex: Path,
    prefer_heuristic: bool = True,
    llm_top_k: int = 8,
) -> Path:
    """
    Backwards-compatible: returns only the path.
    """
    _exam_id, p = pick_reference_with_exam_id(
        bank_dir=bank_dir,
        student_tex=student_tex,
        prefer_heuristic=prefer_heuristic,
        llm_top_k=llm_top_k,
    )
    return p