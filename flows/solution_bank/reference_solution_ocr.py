from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flows.solution_bank.reference_builder import (
    build_reference_bundle_from_mathpix,
    bundle_to_exam_structure,
    questions_bundle_to_tex,
    write_reference_bundle_json,
    write_questions_bundle_json,
)
from schemas.ocr_response import OcrResponse
from schemas.ocr_tasks import ReferenceSolutionOcrResult
from schemas.reference_bundle import ReferenceBundle


_HIGH_CONFIDENCE_VALUES = {"high", "גבוה", "strong"}


def _is_high_confidence_correction(item: Any) -> bool:
    if hasattr(item, "confidence"):
        value = str(item.confidence or "").strip().lower()
    elif isinstance(item, dict):
        value = str(item.get("confidence") or "").strip().lower()
    else:
        value = ""

    return value in _HIGH_CONFIDENCE_VALUES


def build_reference_solution_ocr_result(
    *,
    ocr: OcrResponse,
    questions_bundle: ReferenceBundle | dict[str, Any],
    exam_id: str,
    source_name: str = "",
    out_dir: Path | None = None,
    client: Any = None,
) -> ReferenceSolutionOcrResult:
    """
    Task-specific processor for teacher official-solution uploads.

    Input:
      - OcrResponse from any OCR provider
      - existing questions bundle

    Output:
      ReferenceSolutionOcrResult containing:
        - aligned reference_bundle
        - canonical_tex
        - structure correction counts
    """
    raw_text = ocr.primary_text()
    source_name = source_name or ocr.source_filename

    if isinstance(questions_bundle, ReferenceBundle):
        questions_bundle_model = questions_bundle
    else:
        questions_bundle_model = ReferenceBundle.model_validate(questions_bundle)

    reference_dict = build_reference_bundle_from_mathpix(
        questions_bundle=questions_bundle_model.model_dump(),
        solution_mathpix_text=raw_text,
        source_name=source_name,
        exam_id=exam_id,
        client=client,
    )

    reference_bundle = ReferenceBundle.model_validate(reference_dict)
    reference_for_old_helpers = reference_bundle.model_dump()

    canonical_tex = questions_bundle_to_tex(reference_for_old_helpers)

    corrections = list(reference_bundle.structure_corrections or [])
    high_confidence_corrections = [
        c for c in corrections
        if _is_high_confidence_correction(c)
    ]

    canonical_tex_path = ""

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        reference_bundle_path = out_dir / "reference_bundle.json"
        write_reference_bundle_json(reference_for_old_helpers, reference_bundle_path)

        tex_path = out_dir / "reference_current.tex"
        tex_path.write_text(canonical_tex, encoding="utf-8")
        canonical_tex_path = str(tex_path)

        if high_confidence_corrections:
            corrected_questions_path = out_dir / "questions_only_bundle_corrected_by_reference.json"
            write_questions_bundle_json(reference_for_old_helpers, corrected_questions_path)

            corrected_structure = bundle_to_exam_structure(reference_for_old_helpers)
            corrected_structure_path = out_dir / "exam_structure_corrected_by_reference.json"
            corrected_structure_path.write_text(
                json.dumps(corrected_structure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    warnings = list(reference_bundle.warnings or [])

    return ReferenceSolutionOcrResult(
        ok=True,
        exam_id=exam_id,
        source_name=source_name,
        raw_ocr_text=raw_text,
        reference_bundle=reference_bundle,
        canonical_tex=canonical_tex,
        canonical_tex_path=canonical_tex_path,
        structure_corrections_count=len(corrections),
        high_confidence_corrections_count=len(high_confidence_corrections),
        warnings=warnings,
        needs_teacher_review=bool(warnings or corrections),
        ocr=ocr,
    )