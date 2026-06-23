from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from services.solution_bank.reference_builder import (
    build_questions_only_bundle_from_ocr,
)
from services.solution_bank.full_solution_service import full_solution_to_exam_structure
from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import QuestionsOnlyOcrResult
from schemas.reference_bundle import QuestionsOnlyBundle


def build_questions_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
) -> QuestionsOnlyOcrResult:
    """
    Input:
      OcrResponse from any OCR provider for a teacher questions-only upload.

    Output:
      QuestionsOnlyOcrResult containing:
        - QuestionsOnlyBundle
        - canonical questions-only TeX
        - lightweight exam structure
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    bundle_dict = build_questions_only_bundle_from_ocr(
        ocr_text=raw_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
    )

    questions_bundle = QuestionsOnlyBundle.model_validate(bundle_dict)

    # For questions-only preview, generate a minimal TeX with no solutions.
    from services.solution_bank.reference_builder import questions_bundle_to_tex

    canonical_tex = questions_bundle_to_tex(questions_bundle.model_dump())

    structure = {
        "questions": [
            {
                "question_id": str(q.question_id),
                "parts": [p.part for p in q.parts if p.part],
            }
            for q in questions_bundle.questions
        ]
    }
    structure["question_count"] = len(structure["questions"])
    structure["part_count"] = sum(len(q.get("parts") or []) for q in structure["questions"])

    canonical_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "questions_only_bundle.json"
        bundle_path.write_text(
            json.dumps(questions_bundle.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        tex_path = out_dir / "questions_only_current.tex"
        tex_path.write_text(canonical_tex, encoding="utf-8")
        canonical_tex_path = str(tex_path)

        structure_path = out_dir / "exam_structure.json"
        structure_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    warnings = list(questions_bundle.warnings or [])

    return QuestionsOnlyOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        questions_bundle=questions_bundle,
        canonical_tex=canonical_tex,
        canonical_tex_path=canonical_tex_path,
        exam_structure=structure,
        warnings=warnings,
        needs_teacher_review=bool(warnings),
        ocr=ocr,
    )