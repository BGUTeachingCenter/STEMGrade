from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.ocr_response import OcrResponse
from schemas.reference_bundle import ReferenceBundle


OcrTaskKind = Literal[
    "student_work",
    "questions_only",
    "reference_solution",
    "generic",
]


class OcrTaskInput(BaseModel):
    """
    Shared input wrapper for all OCR task processors.

    Universal OCR already happened before this object is used.
    """

    task_kind: OcrTaskKind
    exam_id: str = ""
    source_name: str = ""

    ocr: OcrResponse

    # For reference-solution alignment, this contains the existing question structure.
    existing_questions_bundle: ReferenceBundle | None = None

    options: dict[str, Any] = Field(default_factory=dict)


class StudentWorkOcrResult(BaseModel):
    """
    Task-specific output for student handwritten / scanned work.

    This is not used by the provider clients.
    This is used after universal OCR.
    """

    schema_version: str = "student_work_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    student_tex: str = ""
    student_tex_path: str = ""

    detected_question_count: int | None = None
    detected_part_count: int | None = None

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = True

    ocr: OcrResponse | None = None


class QuestionsOcrResult(BaseModel):
    """
    Task-specific output for teacher exam/questions upload.
    """

    schema_version: str = "questions_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    questions_bundle: ReferenceBundle

    canonical_tex: str = ""
    canonical_tex_path: str = ""

    exam_structure: dict[str, Any] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False

    ocr: OcrResponse | None = None


class ReferenceSolutionOcrResult(BaseModel):
    """
    Task-specific output for teacher official-solution upload.
    """

    schema_version: str = "reference_solution_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    reference_bundle: ReferenceBundle

    canonical_tex: str = ""
    canonical_tex_path: str = ""

    structure_corrections_count: int = 0
    high_confidence_corrections_count: int = 0

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False

    ocr: OcrResponse | None = None


class GenericOcrTaskResult(BaseModel):
    """
    Fallback result for debug tools or future features.
    """

    schema_version: str = "generic_ocr_task_result_v1"

    ok: bool = True
    task_kind: OcrTaskKind = "generic"
    source_name: str = ""

    raw_ocr_text: str = ""
    ocr: OcrResponse
    warnings: list[str] = Field(default_factory=list)