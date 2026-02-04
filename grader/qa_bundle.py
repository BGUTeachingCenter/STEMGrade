# grader/qa_bundle.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .pdf_cleanse import cleanse_test_pdf

# Try to import your compiler function (module style)
try:
    from .compile_tex import compile_tex_to_pdf  # should exist in grader/compile_tex.py
except Exception:
    compile_tex_to_pdf = None

Key = Tuple[int, str]  # (qnum, part) where part in {"א","ב",""}; "" means no part detected


@dataclass(frozen=True)
class QABundleOutputs:
    bundle_pdf: Path
    bundle_tex: Path
    student_clean_pdf: Path
    reference_clean_pdf: Path
    ref_ranges: Dict[Key, Tuple[int, int]]
    student_ranges: Dict[Key, Tuple[int, int]]
    student_answer_pdfs: Dict[Key, Optional[Path]]


# ============================================================
# Low-level helpers
# ============================================================

def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("\n--- COMMAND FAILED ---")
        print("CMD:", " ".join(cmd))
        print("\nSTDOUT:\n", e.stdout)
        print("\nSTDERR:\n", e.stderr)
        raise


def _normalize_part(p: str) -> str:
    p = (p or "").strip()
    if p in ("a", "A"):
        return "א"
    if p in ("b", "B"):
        return "ב"
    return p


def _compile_tex(tex_path: Path, out_dir: Path, *, font_name: str, passes: int = 2, clean: bool = False, texinputs: list[Path] | None = None) -> Path:
    """
    Prefer grader.compile_tex.compile_tex_to_pdf if available.
    Fall back to xelatex directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if compile_tex_to_pdf is not None:
        # support different signatures
        try:
            pdf = compile_tex_to_pdf(tex_path, out_dir, font_name=font_name, passes=passes, clean=clean, texinputs=texinputs)
        except TypeError:
            try:
                pdf = compile_tex_to_pdf(tex_path, out_dir, font_name=font_name, passes=passes, texinputs=texinputs)
            except TypeError:
                pdf = compile_tex_to_pdf(tex_path, out_dir, texinputs=texinputs)
        # normalize return
        if isinstance(pdf, Path):
            return pdf
        if hasattr(pdf, "pdf"):
            return Path(getattr(pdf, "pdf"))
        return out_dir / f"{tex_path.stem}.pdf"

    # fallback
    for _ in range(passes):
        _run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out_dir}", str(tex_path)])
    pdf_path = out_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"XeLaTeX did not produce {pdf_path}")
    return pdf_path


# ============================================================
# Reference PDF range detection (robust to your PDF style)
# ============================================================

# Style A (your PDF): "(נקודות35) .1" -> qnum=1
_REF_Q_POINTS_DOT_RE = re.compile(r"\(נקודות\s*\d+\)\s*\.?\s*(\d+)", re.UNICODE)
# Style B: "שאלה 1" or "Question 1"
_REF_Q_WORD_RE = re.compile(r"(?:שאלה|Question)\s*(\d+)", re.IGNORECASE | re.UNICODE)

# Part markers: "(א)" or "()א"
_REF_PART_PAREN_RE = re.compile(r"\(\s*([א-תA-Za-z])\s*\)", re.UNICODE)
_REF_PART_EMPTY_PAREN_RE = re.compile(r"\(\s*\)\s*([א-תA-Za-z])", re.UNICODE)


def _extract_qnum(text: str) -> Optional[int]:
    m = _REF_Q_POINTS_DOT_RE.search(text)
    if m:
        return int(m.group(1))
    m = _REF_Q_WORD_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_parts(text: str) -> List[str]:
    parts: List[str] = []
    for m in _REF_PART_EMPTY_PAREN_RE.finditer(text):
        parts.append(_normalize_part(m.group(1)))
    for m in _REF_PART_PAREN_RE.finditer(text):
        parts.append(_normalize_part(m.group(1)))

    parts = [p for p in parts if p in ("א", "ב")]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_reference_ranges(reference_pdf: Path, out_dir: Path) -> Dict[Key, Tuple[int, int]]:
    """
    Build page ranges (1-based inclusive) for each (qnum,part).
    Also writes debug_reference_pages.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(reference_pdf))
    page_keys: List[Optional[Key]] = []
    current_qnum: Optional[int] = None
    current_part: str = ""

    dbg: List[str] = []

    for i in range(doc.page_count):
        text = doc[i].get_text("text") or ""
        dbg.append(f"--- PAGE {i+1} ---")
        dbg.append(text[:2500])
        dbg.append("")

        q = _extract_qnum(text)
        if q is not None:
            current_qnum = q
            current_part = ""  # reset

        if current_qnum is None:
            page_keys.append(None)
            continue

        parts = _extract_parts(text)
        if parts:
            current_part = parts[0]

        page_keys.append((current_qnum, current_part))

    (out_dir / "debug_reference_pages.txt").write_text("\n".join(dbg), encoding="utf-8")

    first_page: Dict[Key, int] = {}
    for idx, k in enumerate(page_keys):
        if k is None:
            continue
        first_page.setdefault(k, idx)

    if not first_page:
        return {}

    ordered = sorted(first_page.items(), key=lambda kv: kv[1])
    ranges: Dict[Key, Tuple[int, int]] = {}
    for j, (key, start0) in enumerate(ordered):
        end0 = (ordered[j + 1][1] - 1) if j + 1 < len(ordered) else (doc.page_count - 1)
        ranges[key] = (start0 + 1, end0 + 1)

    return ranges


# ============================================================
# Student TeX answer parsing (by subsections + parts)
# ============================================================

_TEX_QHDR_RE = re.compile(
    r"\\subsection\*\{[^}]*?(?:Question|שאלה)\s*([0-9]{1,2})[^}]*\}",
    re.UNICODE | re.IGNORECASE
)

# Part markers inside body:
# - \textbf{(א) ...} or \textbf{(א)}
# - or (א) at start of line
_TEX_PART_MARK_RE = re.compile(
    r"(\\textbf\{\(\s*([אבaAbB])\s*\)[^}]*\}|\\textbf\{\(\s*([אבaAbB])\s*\)\}|^\s*\(\s*([אבaAbB])\s*\))",
    re.UNICODE | re.MULTILINE
)


# Fallback format: comment markers like "% Page 5 - Question 1a"
_TEX_PAGEQ_RE = re.compile(
    r"^\s*%\s*Page\s*\d+\s*-\s*Question\s*([0-9]{1,2})\s*([אבaAbB])\s*$",
    re.UNICODE | re.IGNORECASE | re.MULTILINE
)

def _strip_to_solution_block(snippet: str) -> str:
    # Many student templates include the question statement and then a marker like "פתרון:".
    # For bundling we want primarily the student's work, so if we see such a marker,
    # keep only the content after it.
    for tok in ("פתרון:", "Solution:", "solution:"):
        j = snippet.find(tok)
        if j != -1:
            return snippet[j + len(tok):].strip()
    return snippet.strip()



def parse_student_tex_answers(student_tex: Path, out_dir: Path) -> Tuple[Dict[Key, str], Dict[Key, Tuple[int, int]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = student_tex.read_text(encoding="utf-8", errors="replace")

    m_begin = re.search(r"\\begin\{document\}", src)
    m_end = re.search(r"\\end\{document\}", src)
    body = src
    body_offset = 0
    if m_begin and m_end and m_begin.end() < m_end.start():
        body_offset = m_begin.end()
        body = src[m_begin.end():m_end.start()]

    q_matches = list(_TEX_QHDR_RE.finditer(body))
    answers: Dict[Key, str] = {}
    ranges: Dict[Key, Tuple[int, int]] = {}
    dbg: List[str] = []

    if not q_matches:
        # Fallback: some students submit a full document with markers like:
        #   % Page 5 - Question 1a
        #   % Page 7 - Question 1b
        page_hits = list(_TEX_PAGEQ_RE.finditer(body))
        if page_hits:
            dbg.append("Parsed using % Page ... - Question Na/Nb markers.")
            for i, ph in enumerate(page_hits):
                qnum = int(ph.group(1))
                part = _normalize_part(ph.group(2))
                start = ph.end()
                end = page_hits[i + 1].start() if i + 1 < len(page_hits) else len(body)
                block = body[start:end]
                snippet = _strip_to_solution_block(block)
                key = (qnum, part)
                if snippet:
                    answers[key] = snippet
                    ranges[key] = (body_offset + start, body_offset + end)
                    dbg.append(f"== Q{qnum}{part} ==")
                    dbg.append(snippet[:1200])
                    dbg.append("")
            (out_dir / "debug_student_tex_parts.txt").write_text(
                "\n".join(dbg) + "\n\n--- RAW BODY (first 2500 chars) ---\n\n" + body[:2500],
                encoding="utf-8"
            )
            return answers, ranges

        (out_dir / "debug_student_tex_parts.txt").write_text(
            r"No \subsection*{Question N ...} / \subsection*{שאלה N ...} found, and no '% Page ... - Question Na/Nb' markers found.\n\n"
            + body[:2500],
            encoding="utf-8"
        )
        return answers, ranges

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

        # Build split points for א/ב
        split: List[Tuple[str, int]] = []
        for h in hits:
            p = h.group(2) or h.group(3) or h.group(4) or ""
            p = _normalize_part(p)
            if p in ("א", "ב"):
                split.append((p, h.start()))

        # Keep earliest marker for each part
        earliest: Dict[str, int] = {}
        for p, idx in split:
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

            # Strip obvious part label at start
            chunk = re.sub(r"^\s*\\textbf\{\(\s*[אבaAbB]\s*\)[^}]*\}\s*", "", chunk)
            chunk = re.sub(r"^\s*\\textbf\{\(\s*[אבaAbB]\s*\)\}\s*", "", chunk)
            chunk = re.sub(r"^\s*\(\s*[אבaAbB]\s*\)\s*", "", chunk)

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


# ============================================================
# Render student answer snippets to standalone PDFs
# ============================================================

def make_answer_tex(qnum: int, part: str, answer_latex: str, font_name: str) -> str:
    """
    Create a minimal, safe XeLaTeX document for a single answer snippet.
    NOTE: we do NOT trust the snippet to be balanced, but at least errors stay local.
    """
    title = f"Student Answer — Q{qnum}" + (f"({part})" if part else "")
    return (
        r"\documentclass[12pt]{article}" "\n"
        r"\usepackage[a4paper,margin=2cm]{geometry}" "\n"
        r"\usepackage{amsmath,amssymb,mathtools}" "\n"
        r"% ---- Common student macros (robust defaults) ----" "\n"
        r"% Many students use \abs{...}, \norm{...}, \ceil{...}, \floor{...} without defining them." "\n"
        r"% We provide lightweight fallbacks that work with plain amsmath/mathtools." "\n"
        r"\providecommand{\abs}[1]{\left\lvert#1\right\rvert}" "\n"
        r"\providecommand{\norm}[1]{\left\lVert#1\right\rVert}" "\n"
        r"\providecommand{\ceil}[1]{\left\lceil#1\right\rceil}" "\n"
        r"\providecommand{\floor}[1]{\left\lfloor#1\right\rfloor}" "\n"
        r"\providecommand{\set}[1]{\left\{#1\right\}}" "\n"
        r"\providecommand{\paren}[1]{\left(#1\right)}" "\n"
        r"\providecommand{\bracks}[1]{\left[#1\right]}" "\n"
        r"\providecommand{\angles}[1]{\left\langle#1\right\rangle}" "\n"
        r"\providecommand{\RR}{\mathbb{R}}" "\n"
        r"\providecommand{\CC}{\mathbb{C}}" "\n"
        r"\providecommand{\NN}{\mathbb{N}}" "\n"
        r"\providecommand{\ZZ}{\mathbb{Z}}" "\n"
        r"\providecommand{\QQ}{\mathbb{Q}}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\usepackage{fontspec}" "\n"
        r"\usepackage{bidi}" "\n"
        rf"\setmainfont[Script=Hebrew]{{{font_name}}}" "\n"
        rf"\setmonofont{{{font_name}}}" "\n"
        r"\setRTL" "\n"
        r"\setlength{\parskip}{0.6em}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\begin{document}" "\n"
        rf"\section*{{{title}}}" "\n"
        r"\begingroup" "\n"
        + answer_latex.strip()
        + "\n"
        r"\endgroup" "\n"
        r"\end{document}" "\n"
    )


def compile_student_answer_pdfs(
    student_answers: Dict[Key, str],
    out_dir: Path,
    font_name: str,
) -> Dict[Key, Optional[Path]]:
    """
    Compile each answer into its own PDF. Failures return None but do not stop the pipeline.
    """
    ans_dir = out_dir / "student_answers"
    ans_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[Key, Optional[Path]] = {}
    for (qnum, part), snippet in student_answers.items():
        stem = f"student_ans_Q{qnum}" + (f"_{part}" if part else "")
        tex_path = ans_dir / f"{stem}.tex"
        pdf_path = ans_dir / f"{stem}.pdf"

        tex_path.write_text(make_answer_tex(qnum, part, snippet, font_name), encoding="utf-8")

        try:
            compiled_pdf = _compile_tex(tex_path, ans_dir, font_name=font_name, passes=2, clean=False)
            # normalize name
            if compiled_pdf.exists() and compiled_pdf != pdf_path:
                try:
                    compiled_pdf.replace(pdf_path)
                except Exception:
                    pdf_path = compiled_pdf
            out[(qnum, part)] = pdf_path if pdf_path.exists() else compiled_pdf
        except Exception:
            out[(qnum, part)] = None

    return out


# ============================================================
# Bundle generator (includes reference pages + answer PDFs)
# ============================================================

def write_bundle_tex(
    out_tex: Path,
    *,
    reference_pdf: Path,
    ref_ranges: Dict[Key, Tuple[int, int]],
    answer_pdfs: Dict[Key, Optional[Path]],
    font_name: str,
    title: str,
) -> None:
    ref_pdf_tex = str(reference_pdf).replace("\\", "/")

    keys = sorted(ref_ranges.keys(), key=lambda k: (k[0], k[1]))

    tex: List[str] = []
    tex.append(r"\documentclass[12pt]{article}" "\n")
    tex.append(r"\usepackage[a4paper,margin=1.7cm]{geometry}" "\n")
    tex.append(r"\usepackage{hyperref}" "\n")
    tex.append(r"\usepackage{bookmark}" "\n")
    tex.append(r"\usepackage{pdfpages}" "\n")
    tex.append(r"\usepackage{fontspec}" "\n")
    tex.append(r"\usepackage{bidi}" "\n")
    tex.append(rf"\setmainfont[Script=Hebrew]{{{font_name}}}" "\n")
    tex.append(rf"\setmonofont{{{font_name}}}" "\n")
    tex.append(r"\setRTL" "\n")
    tex.append(r"\setlength{\parskip}{0.6em}" "\n")
    tex.append(r"\setlength{\parindent}{0pt}" "\n")
    tex.append(r"\begin{document}" "\n")
    tex.append(rf"\section*{{{title}}}" "\n")
    tex.append(r"\tableofcontents" "\n\newpage" "\n")

    for (qnum, part) in keys:
        start, end = ref_ranges[(qnum, part)]
        sec = f"שאלה {qnum}" + (f"({part})" if part else "")
        tex.append(rf"\section{{{sec}}}" "\n")

        tex.append(r"\subsection*{Reference (question + official solution)}" "\n")
        tex.append(rf"\includepdf[pages={{{start}-{end}}},pagecommand={{}}]{{{ref_pdf_tex}}}" "\n")

        tex.append(r"\subsection*{Student answer (rendered)}" "\n")
        ap = answer_pdfs.get((qnum, part))
        if ap is None:
            # fallback: sometimes we only have (q,"") for a whole question
            ap = answer_pdfs.get((qnum, ""))

        if ap is None:
            tex.append(r"\textcolor{red}{Could not compile student answer for this part.}" "\n")
        else:
            ap_tex = str(ap).replace("\\", "/")
            tex.append(rf"\includepdf[pages=-,pagecommand={{}}]{{{ap_tex}}}" "\n")

        tex.append(r"\newpage" "\n")

    tex.append(r"\end{document}" "\n")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("".join(tex), encoding="utf-8")


# ============================================================
# Public API
# ============================================================

def generate_qa_bundle_pdf(
    *,
    reference_pdf: Path,
    student_tex: Path,
    out_dir: Path,
    bundle_stem: str = "qa_bundle",
    font_name: str = "Arial",
) -> QABundleOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Clean reference/test PDF (remove cover page, formula sheet, trailing blanks)
    cleanse_report = cleanse_test_pdf(reference_pdf, out_dir)
    reference_clean_pdf = cleanse_report.output_pdf

    # 1) Reference ranges
    ref_ranges = find_reference_ranges(reference_clean_pdf, out_dir)
    if not ref_ranges:
        raise RuntimeError(
            "Reference question detection failed.\n"
            "Check build/debug_reference_pages.txt for extracted text.\n"
            "If it is empty, your PDF may be scanned (no selectable text)."
        )

    # 2) Parse student TeX answers
    student_answers, student_ranges = parse_student_tex_answers(student_tex, out_dir)
    if not student_answers:
        raise RuntimeError(
            "Could not parse any student answers from the TeX.\n"
            "Check build/debug_student_tex_parts.txt."
        )

    # 3) Compile full student TeX (useful for validation / later grading)
    student_clean_pdf = _compile_tex(student_tex, out_dir, font_name=font_name, passes=2, clean=True, texinputs=[student_tex.parent])

    # 4) Compile each answer snippet to a PDF (safe embedding)
    answer_pdfs = compile_student_answer_pdfs(student_answers, out_dir, font_name)

    # 5) Build bundle TeX
    bundle_tex = out_dir / f"{bundle_stem}.tex"
    write_bundle_tex(
        bundle_tex,
        reference_pdf=reference_clean_pdf,
        ref_ranges=ref_ranges,
        answer_pdfs=answer_pdfs,
        font_name=font_name,
        title="Q/A Bundle (Reference pages + Rendered student answers)",
    )

    # 6) Compile bundle
    bundle_pdf = _compile_tex(bundle_tex, out_dir, font_name=font_name, passes=2, clean=False)

    # normalize bundle name
    normalized = out_dir / f"{bundle_stem}.pdf"
    if bundle_pdf.exists() and bundle_pdf != normalized:
        try:
            bundle_pdf.replace(normalized)
            bundle_pdf = normalized
        except Exception:
            pass

    if not bundle_pdf.exists():
        raise RuntimeError("Bundle PDF was not created. Check LaTeX logs in build/.")

    return QABundleOutputs(
        bundle_pdf=bundle_pdf,
        bundle_tex=bundle_tex,
        student_clean_pdf=student_clean_pdf,
        reference_clean_pdf=reference_clean_pdf,
        ref_ranges=ref_ranges,
        student_ranges=student_ranges,
        student_answer_pdfs=answer_pdfs,
    )