from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ai_grading.reference_builder import (
    build_questions_bundle_from_mathpix,
    bundle_to_exam_structure,
    questions_bundle_to_tex,
    write_questions_bundle_json,
)
from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import QuestionsOcrResult
from schemas.reference_bundle import ReferenceBundle


def build_questions_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
) -> QuestionsOcrResult:
    """
    Task-specific processor for teacher exam/questions uploads.

    Input:
      OcrResponse from any OCR provider.

    Output:
      QuestionsOcrResult containing:
        - questions_bundle
        - canonical_tex
        - exam_structure

    The existing function name says "mathpix", but after this wrapper it should
    be treated as "OCR text". We can rename it later.
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    bundle_dict = build_questions_bundle_from_mathpix(
        mathpix_text=raw_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
    )

    questions_bundle = ReferenceBundle.model_validate(bundle_dict)
    bundle_for_old_helpers = questions_bundle.model_dump()

    canonical_tex = questions_bundle_to_tex(bundle_for_old_helpers)
    exam_structure = bundle_to_exam_structure(bundle_for_old_helpers)

    canonical_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "questions_only_bundle.json"
        write_questions_bundle_json(bundle_for_old_helpers, bundle_path)

        tex_path = out_dir / "questions_only_current.tex"
        tex_path.write_text(canonical_tex, encoding="utf-8")
        canonical_tex_path = str(tex_path)

        structure_path = out_dir / "exam_structure.json"
        structure_path.write_text(
            __import__("json").dumps(exam_structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    warnings = list(questions_bundle.warnings or [])

    return QuestionsOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        questions_bundle=questions_bundle,
        canonical_tex=canonical_tex,
        canonical_tex_path=canonical_tex_path,
        exam_structure=exam_structure,
        warnings=warnings,
        needs_teacher_review=bool(warnings),
        ocr=ocr,
    )