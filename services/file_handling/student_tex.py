# services/student_tex.py
"""Parse student LaTeX submissions into per-question/per-part answer snippets.

Key rule: part labels are always canonicalized to latin letters: "a", "b", ...
(See `services.part_normalize.normalize_part`.)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from services.file_handling.reference_ranges import Key
from services.file_handling.part_normalize import normalize_part


# Primary format: \subsection*{Question 1 ...} or \subsection*{שאלה 1 ...}
_TEX_QHDR_RE = re.compile(
    r"\\subsection\*\{[^}]*?(?:Question|שאלה)\s*([0-9]{1,2})[^}]*\}",
    re.UNICODE | re.IGNORECASE,
)

# Part markers inside the subsection body:
# - \textbf{(א) ...} / \textbf{(a) ...}
# - \textbf{(א)} / \textbf{(a)}
# - (א) / (a) at start of a line
_TEX_PART_MARK_RE = re.compile(
    r"(\\textbf\{\(\s*([A-Za-zא-ת])\s*\)[^}]*\}|\\textbf\{\(\s*([A-Za-zא-ת])\s*\)\}|^\s*\(\s*([A-Za-zא-ת])\s*\))",
    re.UNICODE | re.MULTILINE,
)

# Fallback format: comment markers like "% Page 5 - Question 1a"
_TEX_PAGEQ_RE = re.compile(
    r"^\s*%\s*Page\s*\d+\s*-\s*Question\s*([0-9]{1,2})\s*([A-Za-zא-ת])\s*$",
    re.UNICODE | re.IGNORECASE | re.MULTILINE,
)


def _strip_to_solution_block(snippet: str) -> str:
    """Heuristic: keep only the student's work portion."""
    for tok in ("פתרון:", "Solution:", "solution:"):
        j = snippet.find(tok)
        if j != -1:
            return snippet[j + len(tok) :].strip()
    return snippet.strip()


def parse_student_tex_answers(student_tex: Path, out_dir: Path) -> Tuple[Dict[Key, str], Dict[Key, Tuple[int, int]]]:
    r"""Parse a student's .tex into answer snippets.

    Returns:
        answers: (qnum, part) -> latex snippet
        ranges : (qnum, part) -> (start_char, end_char) in original file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src = student_tex.read_text(encoding="utf-8", errors="replace")

    # Prefer parsing only the \begin{document}..\end{document} region.
    m_begin = re.search(r"\\begin\{document\}", src)
    m_end = re.search(r"\\end\{document\}", src)
    body = src
    body_offset = 0
    if m_begin and m_end and m_begin.end() < m_end.start():
        body_offset = m_begin.end()
        body = src[m_begin.end() : m_end.start()]

    q_matches = list(_TEX_QHDR_RE.finditer(body))
    answers: Dict[Key, str] = {}
    ranges: Dict[Key, Tuple[int, int]] = {}
    dbg: List[str] = []

    if not q_matches:
        # Fallback parser
        page_hits = list(_TEX_PAGEQ_RE.finditer(body))
        if page_hits:
            dbg.append("Parsed using % Page ... - Question Na/Nb markers.")
            for i, ph in enumerate(page_hits):
                qnum = int(ph.group(1))
                part = normalize_part(ph.group(2))
                start = ph.end()
                end = page_hits[i + 1].start() if i + 1 < len(page_hits) else len(body)
                block = body[start:end]
                snippet = _strip_to_solution_block(block)
                key = (qnum, part)
                if snippet:
                    answers[key] = snippet
                    ranges[key] = (body_offset + start, body_offset + end)
                    dbg.append(f"== Q{qnum}({part}) ==")
                    dbg.append(snippet[:1200])
                    dbg.append("")

            (out_dir / "debug_student_tex_parts.txt").write_text(
                "\n".join(dbg)
                + "\n\n--- RAW BODY (first 2500 chars) ---\n\n"
                + body[:2500],
                encoding="utf-8",
            )
            return answers, ranges

        (out_dir / "debug_student_tex_parts.txt").write_text(
            "No \\subsection*{Question N ...} / \\subsection*{שאלה N ...} found, and no '% Page ... - Question Na/Nb' markers found.\n\n"
            + body[:2500],
            encoding="utf-8",
        )
        return answers, ranges

    # Primary parser: split by subsections
    for qi, qm in enumerate(q_matches):
        qnum = int(qm.group(1))
        q_start = qm.end()
        q_end = q_matches[qi + 1].start() if qi + 1 < len(q_matches) else len(body)
        q_block = body[q_start:q_end]

        hits = list(_TEX_PART_MARK_RE.finditer(q_block))
        if not hits:
            key = (qnum, "")
            snippet = _strip_to_solution_block(q_block)
            if snippet:
                answers[key] = snippet
                ranges[key] = (body_offset + q_start, body_offset + q_end)
                dbg.append(f"== Q{qnum} (no parts) ==")
                dbg.append(snippet[:1200])
                dbg.append("")
            continue

        # Identify split points for parts by earliest marker per part.
        split_points: List[Tuple[str, int]] = []
        for h in hits:
            p = h.group(2) or h.group(3) or h.group(4) or ""
            p = normalize_part(p)
            # keep only single-letter parts like a,b,c...
            if p and len(p) == 1 and p.isalpha():
                split_points.append((p, h.start()))

        earliest: Dict[str, int] = {}
        for p, idx in split_points:
            if p not in earliest or idx < earliest[p]:
                earliest[p] = idx

        if not earliest:
            key = (qnum, "")
            snippet = _strip_to_solution_block(q_block)
            if snippet:
                answers[key] = snippet
                ranges[key] = (body_offset + q_start, body_offset + q_end)
            continue

        points = sorted(earliest.items(), key=lambda x: x[1])

        for pi, (part, start_rel) in enumerate(points):
            end_rel = points[pi + 1][1] if pi + 1 < len(points) else len(q_block)
            chunk = q_block[start_rel:end_rel]

            # Strip obvious part label at the start of the chunk.
            chunk = re.sub(r"^\s*\\textbf\{\(\s*[A-Za-zא-ת]\s*\)[^}]*\}\s*", "", chunk)
            chunk = re.sub(r"^\s*\\textbf\{\(\s*[A-Za-zא-ת]\s*\)\}\s*", "", chunk)
            chunk = re.sub(r"^\s*\(\s*[A-Za-zא-ת]\s*\)\s*", "", chunk)

            chunk = _strip_to_solution_block(chunk)

            key = (qnum, part)
            if chunk:
                answers[key] = chunk
                ranges[key] = (body_offset + q_start + start_rel, body_offset + q_start + end_rel)
                dbg.append(f"== Q{qnum}({part}) ==")
                dbg.append(chunk[:1200])
                dbg.append("")

    (out_dir / "debug_student_tex_parts.txt").write_text("\n".join(dbg), encoding="utf-8")
    return answers, ranges
