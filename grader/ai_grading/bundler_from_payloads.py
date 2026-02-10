# grader/answer_render_payloads.py (or fold into answer_render.py)
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .compile_tex import compile_tex_to_pdf
from .tex_cleaner import clean_tex_robust  # if you want cleaning here

def make_answer_tex(qid: str, answer_latex: str, font_name: str) -> str:
    # Minimal wrapper doc. Keep bidi/RTL consistent.
    return rf"""
\documentclass[12pt]{{article}}
\usepackage[a4paper,margin=1.7cm]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{bidi}}
\setmainfont[Script=Hebrew]{{{font_name}}}
\setRTL
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6em}}
\begin{{document}}
\section*{{{qid}}}
{answer_latex}
\end{{document}}
""".lstrip()

def compile_answer_pdfs_from_payloads(payloads, out_dir: Path, font_name: str) -> Dict[Tuple[int,str], Optional[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_dir = out_dir / "student_answers"
    ans_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for p in payloads:
        key = p.key
        body = (p.student.get("latex_raw") or "").strip()
        if not body:
            results[key] = None
            continue

        tex_path = ans_dir / f"{p.qid.replace('(', '_').replace(')', '')}_answer.tex"
        tex_path.write_text(make_answer_tex(p.qid, body, font_name), encoding="utf-8")

        # IMPORTANT: clean=True for untrusted answer latex
        pdf = compile_tex_to_pdf(tex_path, ans_dir, clean=True, font_name=font_name, texinputs=[ans_dir]).pdf
        results[key] = pdf

    return results
