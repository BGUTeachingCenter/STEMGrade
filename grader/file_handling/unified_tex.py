from __future__ import annotations

from pathlib import Path


def build_unified_tex(
    *,
    qa_tex: Path,
    feedback_tex: Path,
    out_dir: Path,
    output_stem: str = "graded_union",
) -> Path:
    """
    Build one final TeX file by concatenating:
      1) Q/A bundle TeX body
      2) feedback TeX body

    Assumes both inputs are standalone TeX documents and strips the outer
    wrapper so we end up with a single final standalone document.

    This is intentionally simple so it is easy to tweak later.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_text = qa_tex.read_text(encoding="utf-8", errors="replace")
    feedback_text = feedback_tex.read_text(encoding="utf-8", errors="replace")

    qa_body = _extract_document_body(qa_text, source_name=str(qa_tex))
    feedback_body = _extract_document_body(feedback_text, source_name=str(feedback_tex))

    unified = "\n".join([
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
        r"\begin{document}",
        "% ===== BEGIN Q/A BUNDLE =====",
        qa_body.strip(),
        r"\clearpage",
        "% ===== BEGIN FEEDBACK =====",
        feedback_body.strip(),
        r"\end{document}",
        "",
    ])

    out_path = out_dir / f"{output_stem}.tex"
    out_path.write_text(unified, encoding="utf-8")
    return out_path


def _extract_document_body(tex: str, *, source_name: str = "") -> str:
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