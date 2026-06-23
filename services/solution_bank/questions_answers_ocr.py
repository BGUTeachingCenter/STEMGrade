from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import QuestionsAnswersOcrResult
from schemas.reference_bundle import QuestionsAnswersBundle
from services.solution_bank.reference_builder import (
    build_questions_answers_bundle_from_ocr,
    build_questions_answers_bundle_from_questions_only,
    build_questions_only_bundle_from_ocr,
)
from services.solution_bank.full_solution_service import build_full_solution_from_questions_answers


def build_questions_answers_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
    enable_answer_fallback: bool = False,
    answer_fallback_client: Any = None,
    answer_fallback_provider: str = "",
    answer_fallback_model: str = "",
) -> QuestionsAnswersOcrResult:
    """
    Input:
      OcrResponse from any OCR provider for a combined questions+answers upload.

    Output:
      QuestionsAnswersOcrResult containing:
        - QuestionsAnswersBundle
        - promoted FullSolutionBundle
        - canonical full-solution TeX

    Recovery behavior:
      The combined questions+answers builder can be slow/heavy for long PDFs.
      If it fails or times out, recover by building a questions-only bundle from
      the same OCR text, then let the answer fallback generate missing answers.
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    warnings: list[str] = []
    combined_builder_failed = False

    try:
        bundle_dict = build_questions_answers_bundle_from_ocr(
            ocr_text=raw_text,
            source_name=source_name,
            exam_id=exam_id,
            client=client,
        )
    except Exception as exc:
        combined_builder_failed = True
        warning = (
            "Combined questions+answers AI builder failed. "
            "Recovered questions only from OCR and used answer fallback for missing answers. "
            f"Original error: {type(exc).__name__}: {exc}"
        )
        warnings.append(warning)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "questions_answers_ai_builder_error.txt").write_text(
                f"{type(exc).__name__}: {exc}",
                encoding="utf-8",
            )

        questions_only_dict = build_questions_only_bundle_from_ocr(
            ocr_text=raw_text,
            source_name=source_name,
            exam_id=exam_id,
            client=client,
        )

        bundle_dict = build_questions_answers_bundle_from_questions_only(
            questions_bundle=questions_only_dict,
            exam_id=exam_id,
            source_name=source_name,
            warning=warning,
        )

    qa_bundle = QuestionsAnswersBundle.model_validate(bundle_dict)

    should_enable_fallback = bool(enable_answer_fallback or combined_builder_failed)

    full_result = build_full_solution_from_questions_answers(
        questions_answers_bundle=qa_bundle,
        exam_id=exam_id,
        source_name=source_name,
        out_dir=out_dir,
        enable_answer_fallback=should_enable_fallback,
        answer_fallback_client=answer_fallback_client or client,
        answer_fallback_provider=answer_fallback_provider or str(ocr.provider or ""),
        answer_fallback_model=answer_fallback_model or str(ocr.model or ""),
        answer_fallback_raw_text=raw_text,
    )

    warnings = [
        *warnings,
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
        warnings=list(dict.fromkeys([w for w in warnings if str(w).strip()])),
        needs_teacher_review=True if combined_builder_failed else bool(warnings),
        ocr=ocr,
    )