from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from grader.file_handling.part_normalize import normalize_part


@dataclass(frozen=True)
class RefPart:
    """One part of a question, e.g. Q1(a)."""
    qnum: int
    part: str  # canonical: "a", "b", ...
    title: str
    latex_body: str


_SECTION_RE = re.compile(
    r"\\(?:section|subsection)\*\{\s*Question\s+(\d+)\s*\}",
    re.IGNORECASE,
)
# Accepts: \subsection*{(a) ...} or \subsection*{(א) ...}  (with optional title)
_SUBSECTION_RE = re.compile(r"\\subsection\*\{\s*\(([^)]{1,4})\)\s*([^}]*)\}", re.IGNORECASE)


def parse_reference_tex(tex: Union[str, Path]) -> Dict[Tuple[int, str], RefPart]:
    if isinstance(tex, Path):
        tex = tex.read_text(encoding="utf-8", errors="replace")

    tex = tex.replace("\r\n", "\n").replace("\r", "\n")

    sections = list(_SECTION_RE.finditer(tex))
    if not sections:
        raise ValueError(
            "Could not find any \\section*{Question N} blocks in the reference TeX. "
            "Make sure it contains lines like: \\section*{Question 1}"
        )

    results: Dict[Tuple[int, str], RefPart] = {}

    for i, sec in enumerate(sections):
        qnum = int(sec.group(1))
        start = sec.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(tex)
        q_block = tex[start:end]

        subs = list(_SUBSECTION_RE.finditer(q_block))
        if not subs:
            raise ValueError(
                f"Question {qnum} has no \\subsection*{{(a)...}} blocks. "
                "Expected parts like: \\subsection*{(a) ...}"
            )

        for j, sub in enumerate(subs):
            part_letter = normalize_part(sub.group(1))
            title = (sub.group(2) or "").strip()
            body_start = sub.end()
            body_end = subs[j + 1].start() if j + 1 < len(subs) else len(q_block)  # ✅ FIX
            body = q_block[body_start:body_end].strip()

            results[(qnum, part_letter)] = RefPart(
                qnum=qnum,
                part=part_letter,
                title=title,
                latex_body=body,
            )

    return results

