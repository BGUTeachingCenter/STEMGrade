from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import AnswersOnlyOcrResult
from schemas.reference_bundle import AnswersOnlyBundle
from services.solution_bank.reference_builder import build_answers_only_bundle_from_ocr


def build_answers_ocr_result(
    *,
    ocr: OcrResponse,
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
    questions_bundle: dict[str, Any] | None = None,
) -> AnswersOnlyOcrResult:
    """
    Input:
      OcrResponse from any OCR provider for a teacher answers-only upload.

    Output:
      AnswersOnlyOcrResult containing AnswersOnlyBundle.
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    bundle_dict = build_answers_only_bundle_from_ocr(
        ocr_text=raw_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
        questions_bundle=questions_bundle,
    )

    answers_bundle = AnswersOnlyBundle.model_validate(bundle_dict)

    if out_dir is not None:
        import json

        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = out_dir / "answers_only_bundle.json"
        bundle_path.write_text(
            json.dumps(answers_bundle.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    warnings = list(answers_bundle.warnings or [])

    return AnswersOnlyOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        answers_bundle=answers_bundle,
        warnings=warnings,
        needs_teacher_review=bool(warnings),
        ocr=ocr,
    )