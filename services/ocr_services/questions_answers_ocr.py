from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import QuestionsAnswersOcrResult
from schemas.reference_bundle import QuestionsAnswersBundle
from services.ai_grading.reference_builder import build_questions_answers_bundle_from_ocr
from services.ocr_services.full_solution_service import build_full_solution_from_questions_answers


def build_questions_answers_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
) -> QuestionsAnswersOcrResult:
    """
    Input:
      OcrResponse from any OCR provider for a combined questions+answers upload.

    Output:
      QuestionsAnswersOcrResult containing:
        - QuestionsAnswersBundle
        - promoted FullSolutionBundle
        - canonical full-solution TeX
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    bundle_dict = build_questions_answers_bundle_from_ocr(
        ocr_text=raw_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
    )

    qa_bundle = QuestionsAnswersBundle.model_validate(bundle_dict)

    full_result = build_full_solution_from_questions_answers(
        questions_answers_bundle=qa_bundle,
        exam_id=exam_id,
        source_name=source_name,
        out_dir=out_dir,
    )

    warnings = [
        *(qa_bundle.warnings or []),
        *(full_result.warnings or []),
    ]

    return QuestionsAnswersOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        questions_answers_bundle=qa_bundle,
        full_solution_bundle=full_result.full_solution_bundle,
        canonical_tex=full_result.canonical_tex,
        canonical_tex_path=full_result.canonical_tex_path,
        warnings=warnings,
        needs_teacher_review=bool(warnings),
        ocr=ocr,
    )