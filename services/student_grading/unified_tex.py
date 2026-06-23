from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


# Match \section{...} and \section*{...}
_SECTION_RE = re.compile(
    r"\\section\*?\{(?P<title>[^}]*)\}(?P<body>.*?)(?=(\\section\*?\{)|\\end\{document\})",
    re.DOTALL,
)

# Match Q1 or Q1(a)
_QID_RE = re.compile(r"Q\s*(\d+)(?:\(([^)]+)\))?", re.IGNORECASE)

# Inline QA bundle headings like: {\large \textbf{Q1(a)}}\par
_INLINE_QA_HEAD_RE = re.compile(
    r"\{\s*\\large\s+\\textbf\{(?P<label>Q\s*\d+(?:\([^)]+\))?)\}\s*\}\s*\\par",
    re.IGNORECASE,
)

# Inline feedback headings like: {\Large Q1(a)\par}
_INLINE_FB_HEAD_RE = re.compile(
    r"\{\s*\\Large\s+(?P<label>Q\s*\d+(?:\([^)]+\))?)\s*\\par\s*\}",
    re.IGNORECASE,
)

# Markers in inline QA bundle
_INLINE_REF_MARK = r"{\bfseries Reference}\par"
_INLINE_STU_MARK = r"{\bfseries Student answer}\par"


def build_unified_tex(
    *,
    qa_tex: Path,
    feedback_tex: Path,
    out_dir: Path,
    output_stem: str = "graded_union",
) -> Path:
    """
    Build one final TeX file interleaved by question:

      Q:
        1) Student answer
        2) Reference
        3) Feedback

    Supports inputs in TWO styles:
      A) Section style:
         \\section{Q1...} ... \\subsection*{Reference} ... \\subsection*{Student answer} ...
      B) Inline style (your current qa_bundle.tex / feedback.tex):
         \\hrule
         {\\large \\textbf{Q1(a)}}\\par
         {\\bfseries Reference}\\par
         ...
         {\\bfseries Student answer}\\par
         ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_text = qa_tex.read_text(encoding="utf-8", errors="replace")
    fb_text = feedback_tex.read_text(encoding="utf-8", errors="replace")

    qa_body = _extract_document_body(qa_text)
    fb_body = _extract_document_body(fb_text)

    qa_sections = _parse_qa_any(qa_body)
    fb_sections = _parse_feedback_any(fb_body)

    ordered_keys = _ordered_union_keys(qa_sections, fb_sections)

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
        r"\setmainfont{Arial}",
        r"\usepackage{bidi}",
        r"\setRTL",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\begin{document}",
    ]

    if not ordered_keys:
        unified_parts.extend([
            r"\section{Unification failed}",
            r"\textcolor{red}{No question blocks (QIDs) could be parsed from qa\_tex and feedback\_tex.}",
            r"\par",
            r"\textbf{This usually means the parser does not match the bundle/feedback format.}",
            r"\par\medskip",
            r"\textbf{QA body starts with:}\par",
            r"\begin{verbatim}",
            _truncate_for_verbatim(qa_body, 1200),
            r"\end{verbatim}",
            r"\textbf{Feedback body starts with:}\par",
            r"\begin{verbatim}",
            _truncate_for_verbatim(fb_body, 1200),
            r"\end{verbatim}",
            r"\end{document}",
            "",
        ])
        out_path = out_dir / f"{output_stem}.tex"
        out_path.write_text("\n".join(unified_parts), encoding="utf-8")
        return out_path

    for qid in ordered_keys:
        qa = qa_sections.get(qid, {})
        fb = (fb_sections.get(qid) or "").strip()

        section_title = qa.get("title") or qid
        student_block = (qa.get("student") or "").strip()
        reference_block = (qa.get("reference") or "").strip()

        unified_parts.extend([
            f"\\section{{{_tex_escape_title(section_title)}}}",
            r"\subsection*{Student answer}",
            student_block if student_block else r"\textcolor{red}{Missing student answer block.}",
            r"\subsection*{Reference}",
            reference_block if reference_block else r"\textcolor{red}{Missing reference block.}",
            r"\subsection*{Feedback}",
            fb if fb else r"\textcolor{red}{Missing feedback block.}",
            r"\clearpage",
        ])

    unified_parts.append(r"\end{document}")
    unified_parts.append("")

    out_path = out_dir / f"{output_stem}.tex"
    out_path.write_text("\n".join(unified_parts), encoding="utf-8")
    return out_path


def _extract_document_body(tex: str) -> str:
    start_token = r"\begin{document}"
    end_token = r"\end{document}"

    start = tex.find(start_token)
    end = tex.rfind(end_token)

    if start != -1 and end != -1 and end > start:
        return tex[start + len(start_token):end].strip()

    return tex.strip()


# -------------------------
# Parsing QA (either section style or inline style)
# -------------------------

def _parse_qa_any(body: str) -> Dict[str, Dict[str, str]]:
    # Try section-style first
    sections = _parse_qa_sections_section_style(body)
    if sections:
        return sections
    # Fallback: inline-style (qa_bundle.tex)
    return _parse_qa_sections_inline_style(body)


def _parse_qa_sections_section_style(body: str) -> Dict[str, Dict[str, str]]:
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


def _parse_qa_sections_inline_style(body: str) -> Dict[str, Dict[str, str]]:
    """
    Parse inline Q/A bundle:

      \hrule
      {\large \textbf{Q1(a)}}\par
      ...
      {\bfseries Reference}\par
      <reference...>
      ...
      {\bfseries Student answer}\par
      <student...>
      ...
      \hrule (next) OR end
    """
    out: Dict[str, Dict[str, str]] = {}

    matches = list(_INLINE_QA_HEAD_RE.finditer(body))
    if not matches:
        return out

    for i, m in enumerate(matches):
        label = (m.group("label") or "").strip()
        qid = _normalize_qid(label)
        if not qid:
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end]

        ref_block, stu_block = _extract_inline_ref_and_student(chunk)

        out[qid] = {
            "title": qid,
            "reference": ref_block.strip(),
            "student": stu_block.strip(),
        }

    return out


def _extract_inline_ref_and_student(chunk: str) -> Tuple[str, str]:
    """
    Extract reference and student blocks from the inline chunk using the markers:
      {\bfseries Reference}\par
      {\bfseries Student answer}\par
    """
    idx_ref = chunk.find(_INLINE_REF_MARK)
    idx_stu = chunk.find(_INLINE_STU_MARK)

    if idx_ref == -1 and idx_stu == -1:
        # Nothing recognized; return entire chunk as reference to avoid losing data.
        return chunk.strip(), ""

    if idx_ref != -1 and idx_stu != -1 and idx_stu > idx_ref:
        ref_start = idx_ref + len(_INLINE_REF_MARK)
        ref_text = chunk[ref_start:idx_stu].strip()

        stu_start = idx_stu + len(_INLINE_STU_MARK)
        stu_text = chunk[stu_start:].strip()
        return ref_text, stu_text

    # If order is weird, do best effort:
    if idx_ref != -1:
        ref_start = idx_ref + len(_INLINE_REF_MARK)
        return chunk[ref_start:].strip(), ""
    if idx_stu != -1:
        stu_start = idx_stu + len(_INLINE_STU_MARK)
        return "", chunk[stu_start:].strip()

    return "", ""


# -------------------------
# Parsing Feedback (either section style or inline style)
# -------------------------

def _parse_feedback_any(body: str) -> Dict[str, str]:
    # Try section-style first (rare in your current feedback.tex)
    out = _parse_feedback_sections_section_style(body)
    if out:
        return out
    # Fallback: inline-style feedback
    return _parse_feedback_sections_inline_style(body)


def _parse_feedback_sections_section_style(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for match in _SECTION_RE.finditer(body):
        title = (match.group("title") or "").strip()
        qid = _qid_from_title(title)
        if not qid:
            continue
        out[qid] = (match.group("body") or "").strip()
    return out


def _parse_feedback_sections_inline_style(body: str) -> Dict[str, str]:
    """
    Parse inline feedback:

      \hrule ...
      {\Large Q1(a)\par}
      <feedback text ...>
      \hrule (next) OR end
    """
    out: Dict[str, str] = {}

    matches = list(_INLINE_FB_HEAD_RE.finditer(body))
    if matches:
        for i, m in enumerate(matches):
            label = (m.group("label") or "").strip()
            qid = _normalize_qid(label)
            if not qid:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            chunk = body[start:end].strip()
            out[qid] = chunk
        return out

    # Secondary fallback: search for plain "Q1(a)" occurrences at line starts
    qid_heading_re = re.compile(
        r"(?P<head>(?:^|\n)\s*(Q\s*\d+(?:\([^)]+\))?))",
        re.IGNORECASE,
    )
    matches2 = list(qid_heading_re.finditer(body))
    if not matches2:
        return out

    for i, m in enumerate(matches2):
        qid = _normalize_qid(m.group(1))
        if not qid:
            continue
        start = m.start()
        end = matches2[i + 1].start() if i + 1 < len(matches2) else len(body)
        out[qid] = body[start:end].strip()

    return out


# -------------------------
# Helpers
# -------------------------

def _extract_subsection_block(section_body: str, subsection_name: str) -> str:
    pattern = re.compile(
        rf"\\subsection\*\{{{re.escape(subsection_name)}\}}(?P<body>.*?)(?=(\\subsection\*|\\section\*?\{{|$))",
        re.DOTALL,
    )
    match = pattern.search(section_body)
    return (match.group("body") if match else "").strip()


def _qid_from_title(title: str) -> str:
    title = (title or "").strip()

    qid = _normalize_qid(title)
    if qid:
        return qid

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


def _truncate_for_verbatim(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n...\n(TRUNCATED)\n"


def _tex_escape_title(text: str) -> str:
    # safe-ish escaping for section titles
    if not text:
        return ""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )