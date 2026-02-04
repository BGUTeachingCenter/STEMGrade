from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def build_feedback_pdf_from_grades(
    grades_json: Path,
    out_pdf: Path,
    title: str = "Graded Feedback",
    font_path: Path | None = None,
) -> Path:
    """
    Create a feedback PDF from grades.json using ReportLab (no LaTeX).

    If you want Hebrew + Unicode to render reliably, pass a TTF font_path,
    e.g. DejaVuSans.ttf or Arial.ttf (path on your system).
    """
    data = json.loads(grades_json.read_text(encoding="utf-8"))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    font_name = "Helvetica"
    if font_path is not None:
        font_name = "CustomFont"
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4

    # layout
    left = 2.0 * cm
    right = width - 2.0 * cm
    top = height - 2.0 * cm
    y = top

    def draw_line(text: str, size: int = 11, gap: float = 0.55 * cm):
        nonlocal y
        c.setFont(font_name, size)
        # very simple wrapping
        max_width = right - left
        words = text.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font_name, size) <= max_width:
                line = test
            else:
                c.drawString(left, y, line)
                y -= gap
                line = w
                if y < 2.0 * cm:
                    c.showPage()
                    y = top
                    c.setFont(font_name, size)
        if line:
            c.drawString(left, y, line)
            y -= gap
            if y < 2.0 * cm:
                c.showPage()
                y = top

    # Header
    total_score = data.get("total_score", 0)
    total_max = data.get("total_max", 0)

    draw_line(title, size=16, gap=0.8 * cm)
    draw_line(f"Total: {total_score:.1f} / {total_max:.1f}", size=12, gap=0.7 * cm)
    draw_line(" ", size=11, gap=0.4 * cm)

    for q in data.get("questions", []):
        qid = q.get("qid", "Q?")
        score = q.get("score", 0)
        max_points = q.get("max_points", 0)
        summary = q.get("summary", "")

        draw_line("—" * 60, size=10, gap=0.35 * cm)
        draw_line(f"{qid}  ({score:.1f}/{max_points:.1f})", size=13, gap=0.7 * cm)
        if summary:
            draw_line(summary, size=11, gap=0.55 * cm)

        good = q.get("what_was_correct", []) or []
        mistakes = q.get("main_mistakes", []) or []
        improve = q.get("how_to_improve", []) or []

        if good:
            draw_line("What you did well:", size=11, gap=0.55 * cm)
            for item in good[:8]:
                draw_line(f"• {item}", size=11, gap=0.50 * cm)

        if mistakes:
            draw_line("Main mistakes / missing parts:", size=11, gap=0.55 * cm)
            for item in mistakes[:8]:
                draw_line(f"• {item}", size=11, gap=0.50 * cm)

        if improve:
            draw_line("How to improve:", size=11, gap=0.55 * cm)
            for item in improve[:8]:
                draw_line(f"• {item}", size=11, gap=0.50 * cm)

        conf = q.get("confidence", None)
        if conf is not None:
            draw_line(f"Confidence: {float(conf):.2f}", size=10, gap=0.50 * cm)

        draw_line(" ", size=11, gap=0.45 * cm)

    c.save()
    return out_pdf
