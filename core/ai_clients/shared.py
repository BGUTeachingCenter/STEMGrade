from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from schemas.ocr_response import OcrOptions


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

SUPPORTED_OCR_SUFFIXES = frozenset({
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
})


def safe_json_loads(value: str) -> dict[str, Any]:
    """
    Parse a JSON object from a model response.

    Models occasionally wrap JSON in prose or return stray control characters.
    Keep this behavior identical for every provider.
    """
    if not value:
        raise ValueError("Empty model response (expected JSON).")

    cleaned = _CONTROL_CHARS_RE.sub("", value)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = (
        cleaned[start:end + 1]
        if start != -1 and end != -1 and end > start
        else cleaned
    )

    return json.loads(candidate)


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".png":
        return "image/png"

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix == ".webp":
        return "image/webp"

    if suffix == ".pdf":
        return "application/pdf"

    return "application/octet-stream"


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def file_to_data_url(path: Path) -> str:
    return f"data:{guess_mime(path)};base64,{file_to_base64(path)}"


def ensure_supported_ocr_file(
    file_path: Path,
    *,
    provider: str,
) -> str:
    if not file_path.exists():
        raise RuntimeError(
            f"OCR input file does not exist: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_OCR_SUFFIXES:
        raise RuntimeError(
            f"Unsupported {provider} OCR file type: {suffix}"
        )

    return suffix


def build_ocr_prompt(options: OcrOptions) -> str:
    variant = (getattr(options, "prompt_variant", "default") or "default").strip().lower()

    if variant == "student_answer_bundle":
        return f"""
You are a neutral OCR and segmentation engine for handwritten or scanned mathematics homework.

Task: read the uploaded student submission and return JSON only.

Return exactly this JSON shape:
{{
  "student_name": "",
  "student_id": "",
  "exam_id": "",
  "answers": [
    {{
      "question_id": 1,
      "part_key": "a",
      "answer_text": "student's visible written work, preserving mistakes and math notation",
      "visual_elements": [
        "number line: solid segment from -3 to 7 with filled endpoints"
      ],
      "visual_capture_status": "complete",
      "page_numbers": [1],
      "regions": [
        {{
          "page_number": 1,
          "x": 0.10,
          "y": 0.20,
          "width": 0.55,
          "height": 0.18,
          "text_excerpt": "short visible excerpt from this region",
          "confidence": 0.0
        }}
      ],
      "confidence": 0.0,
      "needs_review": false,
      "warnings": []
    }}
  ],
  "unmatched_blocks": [],
  "warnings": []
}}

Rules:
- Preserve the original language. Language hint: {options.language_hint}.
- Use question_id as an integer.
- Convert Hebrew part labels to latin keys: א=a, ב=b, ג=c, ד=d, ה=e, ו=f, ז=g, ח=h, ט=i, י=j.
- If an answer has no visible part label, use part_key "".
- Preserve the student's mistakes. Do not solve, correct, grade, or give feedback.
- Do not invent missing answers.
- Keep each question/part separate. Do not merge Q2 into Q1.
- answer_text must contain all visible written mathematics and prose, but it must not silently omit non-text mathematical work.
- visual_elements must describe every visible mathematical drawing that carries meaning: number lines, graphs, plotted points, shaded regions, geometric diagrams, arrows, sign charts, tables, Venn diagrams, circuit sketches, or labeled figures.
- For a number line, explicitly state the left and right endpoints, whether each endpoint is open or filled/closed, and which interval or rays are drawn.
- For a graph, explicitly state the axes, labeled values, intercepts/key points, open or closed points, shaded regions, and the visible shape or direction.
- For a geometry diagram, explicitly state labels, equal-length marks, angle marks, parallel/perpendicular marks, and any construction used in the solution.
- Use visual_capture_status="complete" only when every visible mathematical drawing was described. Use "none" only when you are confident that the answer contains no mathematical drawing. Otherwise use "partial" or "uncertain", set needs_review=true, and explain why in warnings.
- For every visible answer line or visual element, add a separate tight region whenever practical. Coordinates are normalized to the page: x and y are the top-left corner, width and height are the box size, and every value must be between 0 and 1.
- Use page_number starting at 1. Keep each box tight around the student's answer work; do not include the printed question unless the student wrote inside it.
- If one answer continues in two separated places or on two pages, return multiple regions in reading order.
- text_excerpt must be a short transcription of the visible content inside that region. For visual regions, start it with "[VISUAL]" and summarize the drawing, for example: "[VISUAL] number line, filled endpoints -3 and 7, solid segment between them".
- If a reliable box cannot be determined, use regions: [] and mark needs_review=true with a warning. Never invent coordinates.
- If local numbering is messy, use the visible order and labels, but mark needs_review=true.
- For unclear handwriting, transcribe the best visible reading and set needs_review=true with a warning.
- Put stray visible text that cannot be assigned to a question/part into unmatched_blocks.
- Return JSON only, with no markdown fences and no prose outside JSON.
""".strip()

    document_type = str((options.extra or {}).get("document_type") or "printed").strip().lower()
    upload_role = str((options.extra or {}).get("upload_role") or "generic").strip().lower()
    task_context = str((options.extra or {}).get("task_context") or "ocr").strip().lower()

    if document_type == "handwritten":
        mode_rules = """
Document mode: HANDWRITTEN SCAN.
- Treat the file as handwritten mathematics, possibly on grid paper.
- Read page images visually; do not rely on embedded text.
- Preserve the student's/teacher's original mistakes and uncertain wording.
- Do not clean up the mathematics into a corrected solution.
- Mark unclear words or symbols as [unclear] rather than guessing.
- Preserve visible question numbers and Hebrew part labels such as א, ב, ג.
- Keep answer/question boundaries explicit when visible.
""".strip()
    else:
        mode_rules = """
Document mode: PRINTED / TYPED DOCUMENT.
- Treat the file as a printed or typed mathematics document.
- Preserve printed layout, question numbers, headings, and part labels.
- If the PDF has selectable text, keep that text faithfully.
- Do not add interpretation beyond the visible content.
""".strip()

    role_rules = """
Teacher solution-bank context:
- The file may be questions-only, answers-only, or combined questions+answers.
- Preserve enough numbering and labels so a later structuring step can build full_solution_bundle.json.
- Do not decide grading, do not summarize, and do not add feedback.
""".strip() if "teacher_solution_bank" in task_context or upload_role in {"questions_only", "answers_only", "questions_answers"} else ""

    return f"""
You are a neutral OCR engine for mathematics documents.

Extract the visible text from the uploaded file.

{mode_rules}

{role_rules}

Requirements:
- Preserve the original language. Language hint: {options.language_hint}.
- Preserve Hebrew text when visible.
- Preserve question numbers and part labels such as א, ב, ג, a, b, c.
- Preserve mathematical notation using LaTeX where useful.
- Preserve line breaks when they help structure.
- Do not solve.
- Do not correct mathematical mistakes.
- Do not summarize.
- Do not add explanations.
- Return only the extracted OCR text.
""".strip()
