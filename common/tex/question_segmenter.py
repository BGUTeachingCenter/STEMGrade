"""
Format-agnostic segmentation of teacher exam/solution documents into questions
and parts.

Teacher uploads come in many shapes:

  * LaTeX with ``\\section*{שאלה 1}`` / ``\\subsection*{(א) ...}``
  * LaTeX with ``\\section*{Question 1}`` / ``\\subsection*{(a) ...}``
  * Plain text / OCR with lines like ``שאלה 1``, ``Question 1``, ``בעיה 1``
  * Bare numbering like ``1.`` / ``1)`` / ``(1)`` and parts ``(א)`` / ``א)`` / ``(a)``

Relying on a single header style (or on the LLM) to detect structure is
fragile. This module tries several detectors in priority order and picks the
one that best fits the document, then slices the text into per-question and
per-part blocks. Every question is guaranteed to yield at least one part, so a
downstream bundle never ends up with zero parts.

The output is intentionally simple (dataclasses); callers map it onto their own
schema (QuestionsAnswersBundle, QuestionsOnlyBundle, ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.tex.part_normalize import normalize_part


# Ordered Hebrew part letters (א=1 ... י=10) used for sequence validation.
_HEB_LETTERS = "אבגדהוזחטיכ"
_LAT_LETTERS = "abcdefghijk"

# Words that introduce a solution/answer inside a part body.
_SOLUTION_MARKERS = [
    r"\\solution\b",
    r"\\proofword\b",
    r"\\answer\b",
    r"פתרון",
    r"הוכחה",
    r"\bsolution\b",
    r"\bproof\b",
]
_SOLUTION_MARKER_RE = re.compile("|".join(_SOLUTION_MARKERS), re.IGNORECASE)

# Markers that a section is optional / not for submission.
_NON_SUBMISSION_MARKERS = [
    "שאלות שאינן להגשה",
    "שאלות רשות",
    "שאלות בונוס",
    "לא להגשה",
    "אינו להגשה",
    "אינה להגשה",
    "אופציונלי",
    "not for submission",
    "optional",
    "bonus",
]


@dataclass
class SegPart:
    label: str            # display label as seen, e.g. "א", "a", "1", or ""
    part_key: str         # canonical latin key: "a", "b", ...
    question_text: str    # the prompt text for this part (stem + part prompt)
    official_solution: str
    expected_answer: str = ""


@dataclass
class SegQuestion:
    question_id: int
    display_id: str
    parts: list[SegPart] = field(default_factory=list)
    section_label: str = ""
    is_gradeable: bool = True


# ---------------------------------------------------------------------
# Question-header detection
# ---------------------------------------------------------------------

# Each detector returns match objects whose group "num" (if present) is the
# visible question number. Ordered by how specific/reliable they are.
_QUESTION_DETECTORS: list[re.Pattern[str]] = [
    # \section*{שאלה 1} / \section{Question 1} / \subsection*{שאלה 1}
    re.compile(
        r"(?m)^[ \t]*\\(?:sub)?section\*?\s*\{\s*"
        r"(?:שאלה|בעיה|תרגיל|question|exercise|problem|q)\s*"
        r"(?P<num>\d{1,3})\s*\*?[^}]*\}",
        re.IGNORECASE,
    ),
    # \section*{1} / \section*{1.} — a bare number inside a section
    re.compile(
        r"(?m)^[ \t]*\\(?:sub)?section\*?\s*\{\s*(?P<num>\d{1,3})[.)\s]*\}",
    ),
    # Plain text header: שאלה 1 / בעיה 1 / תרגיל 1 / Question 1 / Q1 / Problem 1
    re.compile(
        r"(?m)^[ \t]*(?:שאלה|בעיה|תרגיל|question|exercise|problem|q)\s*"
        r"(?P<num>\d{1,3})\s*\*?\s*[.:)]?",
        re.IGNORECASE,
    ),
    # Bare numbered lines: 1. / 1) / (1)  (lowest priority — noisiest)
    re.compile(r"(?m)^[ \t]*\(?(?P<num>\d{1,3})[.)]\s"),
]


def _non_submission_index(text: str) -> int | None:
    lowered = text.lower()
    best: int | None = None
    for marker in _NON_SUBMISSION_MARKERS:
        idx = lowered.find(marker.lower())
        if idx >= 0 and (best is None or idx < best):
            best = idx
    return best


def _choose_question_detector(text: str) -> list[re.Match[str]]:
    """Pick the best question detector for this document.

    Prefer the highest-priority detector that finds at least two headers; if
    none finds two, use whichever found the most (>=1). This lets a clean
    LaTeX file, a plain-text OCR file, and a bare-numbered file all work.
    """
    best_matches: list[re.Match[str]] = []
    for detector in _QUESTION_DETECTORS:
        matches = list(detector.finditer(text))
        if len(matches) >= 2:
            return matches
        if len(matches) > len(best_matches):
            best_matches = matches
    return best_matches


# ---------------------------------------------------------------------
# Part-header detection
# ---------------------------------------------------------------------

_PART_DETECTORS: list[re.Pattern[str]] = [
    # \subsection*{(א) ...} / \subsection*{(a) ...} / \subsection*{א}
    re.compile(
        r"(?m)^[ \t]*\\(?:sub)*section\*?\s*\{\s*\(?\s*"
        r"(?P<label>[א-יa-zA-Z])\s*\)?[^}]*\}",
    ),
    # Line-leading parenthesised label: (א) / (a)
    re.compile(r"(?m)^[ \t]*\(\s*(?P<label>[א-יa-z])\s*\)"),
    # Line-leading label with delimiter: א) / א. / a) / a.
    re.compile(r"(?m)^[ \t]*(?P<label>[א-יa-z])\s*[.)]\s"),
    # "סעיף א" / "part a"
    re.compile(r"(?m)^[ \t]*(?:סעיף|part)\s*(?P<label>[א-יa-z])\b", re.IGNORECASE),
]


def _label_ordinal(label: str) -> int | None:
    key = normalize_part(label)
    if len(key) == 1 and key in _LAT_LETTERS:
        return _LAT_LETTERS.index(key)
    return None


def _looks_sequential(labels: list[str]) -> bool:
    """Accept part labels only if they look like a real a,b,c / א,ב,ג sequence.

    Guards against stray single Hebrew letters in ordinary math text being
    mistaken for part markers.
    """
    ordinals = [_label_ordinal(x) for x in labels]
    if any(o is None for o in ordinals):
        return False
    # First label should be near the start (a/b), and the sequence should be
    # non-decreasing without large gaps.
    if ordinals[0] > 1:
        return False
    for prev, cur in zip(ordinals, ordinals[1:]):
        if cur <= prev or cur - prev > 1:
            return False
    return True


def _choose_part_detector(block: str) -> list[re.Match[str]]:
    for detector in _PART_DETECTORS:
        matches = list(detector.finditer(block))
        if len(matches) < 1:
            continue
        labels = [m.group("label") for m in matches]
        # A single subsection-style match is trustworthy; multiple inline
        # matches must look like a real sequence.
        if len(matches) == 1 or _looks_sequential(labels):
            return matches
    return []


# ---------------------------------------------------------------------
# Content slicing
# ---------------------------------------------------------------------

def _split_solution(body: str) -> tuple[str, str]:
    """Split a part/question body into (prompt_text, solution_text).

    Uses the first solution marker (\\solution, פתרון, ...) as the boundary.
    If no marker is present, the whole body is treated as solution content and
    the prompt is left empty (the caller may still have a stem).
    """
    m = _SOLUTION_MARKER_RE.search(body)
    if not m:
        return "", body.strip()
    prompt = body[: m.start()].strip()
    solution = body[m.end():].strip()
    return prompt, solution


def _extract_expected_answer(text: str) -> str:
    """Best-effort short answer: prefer \\boxed{...}, else after תשובה/answer."""
    boxed = re.search(r"\\boxed\{", text)
    if boxed:
        # Balance braces from the opening of \boxed{ ... }.
        i = boxed.end()
        depth = 1
        out = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
            i += 1
        inner = "".join(out).strip()
        if inner:
            return inner
    m = re.search(r"(?:תשובה|answer)\s*[:.]?\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return ""


def _clean_fragment(text: str) -> str:
    """Trim common LaTeX section/subsection headers left at the edges."""
    text = re.sub(r"^\s*\\(?:sub)*section\*?\s*\{[^}]*\}", "", text, count=1)
    return text.strip()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def segment_document(text: str) -> list[SegQuestion]:
    """Segment a teacher document into questions and parts.

    Returns an empty list when no question structure can be detected (the
    caller should then fall back to an LLM-based builder).
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    q_matches = _choose_question_detector(text)
    if not q_matches:
        return []

    non_sub_idx = _non_submission_index(text)

    questions: list[SegQuestion] = []
    seen_ids: set[int] = set()

    for i, qm in enumerate(q_matches):
        start = qm.end()
        end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)
        block = text[start:end]
        header_pos = qm.start()

        try:
            qid = int(qm.group("num"))
        except (ValueError, IndexError, TypeError):
            qid = i + 1

        # De-duplicate ids while keeping order (starred/non-submission questions
        # sometimes restart numbering).
        base_qid = qid
        while qid in seen_ids:
            qid += 1000  # push into a separate band rather than collide
        seen_ids.add(qid)

        is_gradeable = non_sub_idx is None or header_pos < non_sub_idx
        section_label = "" if is_gradeable else "non-submission"

        parts = _segment_parts(block)

        questions.append(
            SegQuestion(
                question_id=qid,
                display_id=str(base_qid),
                parts=parts,
                section_label=section_label,
                is_gradeable=is_gradeable,
            )
        )

    return questions


def _segment_parts(block: str) -> list[SegPart]:
    """Segment one question block into parts, guaranteeing at least one part."""
    p_matches = _choose_part_detector(block)

    # Text before the first part header is the shared stem.
    if p_matches:
        stem = _clean_fragment(block[: p_matches[0].start()])
    else:
        stem = _clean_fragment(block)

    parts: list[SegPart] = []

    if not p_matches:
        # Single implicit part carrying the whole question body.
        prompt, solution = _split_solution(stem)
        question_text = prompt or stem
        parts.append(
            SegPart(
                label="",
                part_key="a",
                question_text=question_text.strip(),
                official_solution=solution,
                expected_answer=_extract_expected_answer(block),
            )
        )
        return parts

    for j, pm in enumerate(p_matches):
        p_start = pm.end()
        p_end = p_matches[j + 1].start() if j + 1 < len(p_matches) else len(block)
        body = block[p_start:p_end].strip()

        label = pm.group("label")
        prompt, solution = _split_solution(body)

        # Compose the part's question text: shared stem + this part's prompt.
        question_text_parts = [t for t in (stem, prompt) if t]
        question_text = "\n\n".join(question_text_parts).strip()

        # Canonical key from the label; fall back to positional a,b,c if the
        # label doesn't normalize to a clean letter.
        part_key = normalize_part(label)
        if not part_key and j < len(_LAT_LETTERS):
            part_key = _LAT_LETTERS[j]

        parts.append(
            SegPart(
                label=label,
                part_key=part_key,
                question_text=question_text,
                official_solution=solution,
                expected_answer=_extract_expected_answer(body),
            )
        )

    return parts
