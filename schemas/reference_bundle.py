from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReviewStatus = Literal["ok", "needs_review", "missing", "conflict", "uncertain"]


class StructureCorrection(BaseModel):
    """
    A correction suggested when the solution/reference file proves
    that the original question OCR structure was wrong.
    """

    type: str = ""
    description: str = ""
    confidence: str = ""

    question_id: str = ""
    part: str = ""

    old_text: str = ""
    new_text: str = ""

    needs_teacher_review: bool = True


class ReferencePart(BaseModel):
    """
    One gradeable part, for example Q1א.
    This is used both for questions-only bundles and full reference bundles.
    """

    part: str = ""
    part_key: str = ""

    question_text: str = ""
    required_action: str = ""

    official_solution: str = ""
    expected_answer: str = ""
    grading_instructions: str = ""

    max_points: float | None = None

    review_status: ReviewStatus = "ok"
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ReferenceQuestion(BaseModel):
    question_id: int
    parts: list[ReferencePart] = Field(default_factory=list)


class ReferenceBundle(BaseModel):
    """
    Canonical MathGrade question/reference bundle.

    Questions-only upload:
      official_solution / expected_answer / grading_instructions may be empty.

    Reference upload:
      those fields should be filled where possible.
    """

    schema_version: str = "reference_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[ReferenceQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)

    def to_json_schema_for_openai(self) -> dict[str, Any]:
        """
        Strict-ish JSON schema for OpenAI structured outputs.
        """
        return self.model_json_schema()


def reference_bundle_schema() -> dict[str, Any]:
    return ReferenceBundle.model_json_schema()