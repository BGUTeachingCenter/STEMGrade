from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.ocr_response import OcrResponse
from schemas.reference_bundle import (
    AnswersOnlyBundle,
    FullSolutionBundle,
    QuestionsAnswersBundle,
    QuestionsOnlyBundle,
    ReferenceBundle,
)


OcrTaskKind = Literal[
    "student_work",
    "questions_only",
    "answers_only",
    "questions_answers",
    "full_solution",
    "reference_solution",
    "generic",
]


class OcrTaskInput(BaseModel):
    """
    Input:
      Shared wrapper passed into OCR-service functions after universal OCR.

    Output:
      Not usually stored directly. The task service converts this into one of:
        - StudentWorkOcrResult
        - QuestionsOnlyOcrResult
        - AnswersOnlyOcrResult
        - QuestionsAnswersOcrResult
        - FullSolutionBuildResult
    """

    task_kind: OcrTaskKind
    exam_id: str = ""
    source_name: str = ""

    ocr: OcrResponse

    existing_questions_bundle: QuestionsOnlyBundle | ReferenceBundle | None = None
    existing_answers_bundle: AnswersOnlyBundle | None = None

    options: dict[str, Any] = Field(default_factory=dict)


class StudentWorkOcrResult(BaseModel):
    """
    Input:
      OCR from a student handwritten/scanned upload.

    Output:
      Reviewable student TeX that can later be parsed by the grading pipeline.
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


class QuestionsOnlyOcrResult(BaseModel):
    """
    Input:
      OCR from a teacher uploaded questions-only file.

    Output:
      QuestionsOnlyBundle plus canonical questions-only TeX and exam structure.
    """

    schema_version: str = "questions_only_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    questions_bundle: QuestionsOnlyBundle

    canonical_tex: str = ""
    canonical_tex_path: str = ""

    exam_structure: dict[str, Any] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False

    ocr: OcrResponse | None = None


class AnswersOnlyOcrResult(BaseModel):
    """
    Input:
      OCR from a teacher uploaded answers-only / official-solution file.

    Output:
      AnswersOnlyBundle. It is not fully gradeable until merged with questions.
    """

    schema_version: str = "answers_only_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    answers_bundle: AnswersOnlyBundle

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False

    ocr: OcrResponse | None = None


class QuestionsAnswersOcrResult(BaseModel):
    """
    Input:
      OCR from a teacher uploaded combined questions+answers file.

    Output:
      QuestionsAnswersBundle and optionally a promoted FullSolutionBundle.
    """

    schema_version: str = "questions_answers_ocr_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    raw_ocr_text: str = ""

    questions_answers_bundle: QuestionsAnswersBundle
    full_solution_bundle: FullSolutionBundle | None = None

    canonical_tex: str = ""
    canonical_tex_path: str = ""

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False

    ocr: OcrResponse | None = None


class FullSolutionBuildResult(BaseModel):
    """
    Input:
      Either:
        - QuestionsOnlyBundle + AnswersOnlyBundle
        - QuestionsAnswersBundle
        - ReferenceBundle legacy data

    Output:
      FullSolutionBundle and canonical full solution TeX.
    """

    schema_version: str = "full_solution_build_result_v1"

    ok: bool = True
    exam_id: str = ""
    source_name: str = ""

    full_solution_bundle: FullSolutionBundle

    canonical_tex: str = ""
    canonical_tex_path: str = ""

    q_count: int | None = None
    part_count: int | None = None

    warnings: list[str] = Field(default_factory=list)
    needs_teacher_review: bool = False


class ReferenceSolutionOcrResult(BaseModel):
    """
    Backward-compatible result name.

    Input:
      OCR from a teacher uploaded official-solution file plus existing questions.

    Output:
      Legacy ReferenceBundle and canonical TeX.

    Later this should be replaced by:
      AnswersOnlyOcrResult + FullSolutionBuildResult
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
    Input:
      OCR from any unsupported/debug task.

    Output:
      Raw OCR text plus provider-neutral OCR response.
    """

    schema_version: str = "generic_ocr_task_result_v1"

    ok: bool = True
    task_kind: OcrTaskKind = "generic"
    source_name: str = ""

    raw_ocr_text: str = ""
    ocr: OcrResponse
    warnings: list[str] = Field(default_factory=list)


# Backward-compatible alias for old imports.
QuestionsOcrResult = QuestionsOnlyOcrResult