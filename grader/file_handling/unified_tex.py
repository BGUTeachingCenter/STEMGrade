from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


_SECTION_RE = re.compile(
    r"\\section\{(?P<title>[^}]*)\}(?P<body>.*?)(?=(\\section\{)|\\end\{document\})",
    re.DOTALL,
)

_QID_RE = re.compile(r"Q\s*(\d+)(?:\(([^)]+)\))?", re.IGNORECASE)


def build_unified_tex(
    *,
    qa_tex: Path,
    feedback_tex: Path,
    out_dir: Path,
    output_stem: str = "graded_union",
) -> Path:
    """
    Build one final TeX file interleaved by question:

      Question N:
        1) Student answer
        2) Reference
        3) Feedback

    Expected inputs:
      - qa_tex: a standalone TeX document whose body contains per-question
        \\section{...} blocks with "Reference" and "Student answer" subsections.
      - feedback_tex: a standalone TeX document whose body contains per-question
        feedback blocks keyed by Q1 / Q1(a) / etc.

    Output:
      - one standalone TeX file in out_dir / f"{output_stem}.tex"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_text = qa_tex.read_text(encoding="utf-8", errors="replace")
    feedback_text = feedback_tex.read_text(encoding="utf-8", errors="replace")

    qa_body = _extract_document_body(qa_text)
    feedback_body = _extract_document_body(feedback_text)

    qa_sections = _parse_qa_sections(qa_body)
    feedback_sections = _parse_feedback_sections(feedback_body)

    ordered_keys = _ordered_union_keys(qa_sections, feedback_sections)

    unified_parts: List[str] = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{enumitem}",
        r"\usepackage{fontspec}",
        r"\usepackage{bidi}",
        r"\setRTL",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\begin{document}",
    ]

    for qid in ordered_keys:
        qa = qa_sections.get(qid, {})
        feedback = feedback_sections.get(qid, "").strip()

        section_title = qa.get("title") or qid
        student_block = (qa.get("student") or "").strip()
        reference_block = (qa.get("reference") or "").strip()

        unified_parts.extend([
            f"\\section{{{section_title}}}",
            r"\subsection*{Student answer}",
            student_block if student_block else r"\textcolor{red}{Missing student answer block.}",
            r"\subsection*{Reference}",
            reference_block if reference_block else r"\textcolor{red}{Missing reference block.}",
            r"\subsection*{Feedback}",
            feedback if feedback else r"\textcolor{red}{Missing feedback block.}",
            r"\clearpage",
        ])

    unified_parts.append(r"\end{document}")
    unified_parts.append("")

    out_path = out_dir / f"{output_stem}.tex"
    out_path.write_text("\n".join(unified_parts), encoding="utf-8")
    return out_path


def _extract_document_body(tex: str) -> str:
    """
    Extract content between \\begin{document} and \\end{document}.
    If not found, return the original text stripped.
    """
    start_token = r"\begin{document}"
    end_token = r"\end{document}"

    start = tex.find(start_token)
    end = tex.rfind(end_token)

    if start != -1 and end != -1 and end > start:
        return tex[start + len(start_token):end].strip()

    return tex.strip()


def _parse_qa_sections(body: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the Q/A bundle into:
      {
        "Q1": {
          "title": "שאלה 1" or "Q1(a)",
          "student": "...",
          "reference": "..."
        },
        ...
      }
    """
    out: Dict[str, Dict[str, str]] = {}

    for match in _SECTION_RE.finditer(body):
        title = (match.group("title") or "").strip()
        section_body = (match.group("body") or "").strip()

        qid = _qid_from_title(title)
        if not qid:
            continue

        reference_block = _extract_subsection_block(section_body, "Reference")
        student_block = _extract_subsection_block(section_body, "Student answer")

        out[qid] = {
            "title": title,
            "student": student_block.strip(),
            "reference": reference_block.strip(),
        }

    return out


def _parse_feedback_sections(body: str) -> Dict[str, str]:
    """
    Parse feedback TeX into per-question blocks.

    Strategy:
    - split by \\section{...} first if present and qid-like
    - otherwise fall back to searching for Q1 / Q1(a) headings inside the body
    """
    out: Dict[str, str] = {}

    section_matches = list(_SECTION_RE.finditer(body))
    found_any_section_qid = False

    for match in section_matches:
        title = (match.group("title") or "").strip()
        qid = _qid_from_title(title)
        if not qid:
            continue
        found_any_section_qid = True
        out[qid] = (match.group("body") or "").strip()

    if found_any_section_qid:
        return out

    # Fallback: split on visible Q identifiers inside the feedback body.
    qid_heading_re = re.compile(
        r"(?P<head>(?:^|\n)\s*(?:\\textbf\{)?\s*(Q\s*\d+(?:\([^)]+\))?)\s*(?:\})?)",
        re.IGNORECASE,
    )

    matches = list(qid_heading_re.finditer(body))
    if not matches:
        return out

    for i, match in enumerate(matches):
        qid_raw = match.group(2)
        qid = _normalize_qid(qid_raw)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if qid:
            out[qid] = chunk

    return out


def _extract_subsection_block(section_body: str, subsection_name: str) -> str:
    """
    Extract content under:
      \\subsection*{<subsection_name>}
    until the next subsection/section/end.
    """
    pattern = re.compile(
        rf"\\subsection\*\{{{re.escape(subsection_name)}\}}(?P<body>.*?)(?=(\\subsection\*|\\section\{{|$))",
        re.DOTALL,
    )
    match = pattern.search(section_body)
    return (match.group("body") if match else "").strip()


def _qid_from_title(title: str) -> str:
    """
    Convert titles like:
      'Q1'
      'Q1(a)'
      'שאלה 1'
      'שאלה 1(a)'
    into normalized keys like:
      'Q1'
      'Q1(a)'
    """
    title = (title or "").strip()

    # direct Q...
    qid = _normalize_qid(title)
    if qid:
        return qid

    # Hebrew title like "שאלה 1" or "שאלה 1(a)"
    heb_match = re.search(r"שאלה\s*(\d+)(?:\(([^)]+)\))?", title)
    if heb_match:
        qnum = heb_match.group(1)
        part = (heb_match.group(2) or "").strip()
        return f"Q{qnum}({part})" if part else f"Q{qnum}"

    return ""


def _normalize_qid(text: str) -> str:
    match = _QID_RE.search(text or "")
    if not match:
        return ""

    qnum = match.group(1)
    part = (match.group(2) or "").strip()
    return f"Q{qnum}({part})" if part else f"Q{qnum}"


def _ordered_union_keys(
    qa_sections: Dict[str, Dict[str, str]],
    feedback_sections: Dict[str, str],
) -> List[str]:
    keys = set(qa_sections.keys()) | set(feedback_sections.keys())

    def sort_key(qid: str) -> Tuple[int, str]:
        m = _QID_RE.search(qid)
        if not m:
            return (10**9, qid)
        qnum = int(m.group(1))
        part = (m.group(2) or "").strip()
        return (qnum, part)

    return sorted(keys, key=sort_key)