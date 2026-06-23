from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReviewStatus = Literal["ok", "needs_review", "missing", "conflict", "uncertain"]


# ---------------------------------------------------------------------
# Shared correction / warning structures
# ---------------------------------------------------------------------


class StructureCorrection(BaseModel):
    """
    Input:
      Produced by AI services when OCR/question structure appears wrong.

    Output:
      Stored inside questions/answers/full-solution bundles so the teacher
      and later services can see what was corrected or what needs review.
    """

    type: str = ""
    description: str = ""
    confidence: str = ""

    question_id: str = ""
    part: str = ""

    old_text: str = ""
    new_text: str = ""

    needs_teacher_review: bool = True


# ---------------------------------------------------------------------
# 1. Questions-only schema
# ---------------------------------------------------------------------


class QuestionsOnlyPart(BaseModel):
    """
    Input:
      A teacher uploaded exam/questions file, after OCR and AI structuring.

    Output:
      One gradeable question part with only the question text and required task.
      No official answer is required here.
    """

    part: str = ""
    part_key: str = ""

    question_text: str = ""
    required_action: str = ""

    max_points: float | None = None

    review_status: ReviewStatus = "ok"
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class QuestionsOnlyQuestion(BaseModel):
    """
    Input:
      Group of parts belonging to one detected question number.

    Output:
      Stable question skeleton used later to align answers-only files.
    """

    question_id: int
    parts: list[QuestionsOnlyPart] = Field(default_factory=list)


class QuestionsOnlyBundle(BaseModel):
    """
    Input:
      OCR text from a teacher-uploaded questions-only exam file.

    Output:
      Canonical structured exam skeleton:
        - question IDs
        - part labels
        - question text
        - required student action

      This file is not enough for grading until answers are added.
    """

    schema_version: str = "questions_only_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[QuestionsOnlyQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# 2. Answers-only schema
# ---------------------------------------------------------------------


class AnswersOnlyPart(BaseModel):
    """
    Input:
      A teacher uploaded answers/solutions-only file, after OCR and AI extraction.

    Output:
      One extracted answer/solution part.

    Notes:
      Because the file may not contain full question text, question_id and part
      may be inferred from headings, order, or nearby labels. If uncertain,
      review_status should be "needs_review" or "uncertain".
    """

    question_id: int | None = None

    part: str = ""
    part_key: str = ""

    official_solution: str = ""
    expected_answer: str = ""
    grading_instructions: str = ""

    # Optional hint if the answer file includes a short question reference.
    question_hint: str = ""

    max_points: float | None = None

    review_status: ReviewStatus = "ok"
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class AnswersOnlyQuestion(BaseModel):
    """
    Input:
      Group of answer parts detected under one question number.

    Output:
      Structured answers that can later be merged with QuestionsOnlyBundle.
    """

    question_id: int | None = None
    parts: list[AnswersOnlyPart] = Field(default_factory=list)


class AnswersOnlyBundle(BaseModel):
    """
    Input:
      OCR text from a teacher-uploaded answers-only / official-solutions file.

    Output:
      Structured answers without necessarily containing full question text.

    This file becomes gradeable only after merging with:
      - an existing QuestionsOnlyBundle, or
      - question text recovered from the answers file with high confidence.
    """

    schema_version: str = "answers_only_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[AnswersOnlyQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# 3. Questions + answers schema
# ---------------------------------------------------------------------


class QuestionsAnswersPart(BaseModel):
    """
    Input:
      A combined teacher upload containing both questions and answers.

    Output:
      One part containing question text and answer/solution fields together.
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


class QuestionsAnswersQuestion(BaseModel):
    """
    Input:
      One detected question from a combined questions+answers file.

    Output:
      Parts that can be converted directly into a FullSolutionBundle.
    """

    question_id: int
    parts: list[QuestionsAnswersPart] = Field(default_factory=list)


class QuestionsAnswersBundle(BaseModel):
    """
    Input:
      OCR text from a teacher-uploaded file that contains both exam questions
      and official answers/solutions.

    Output:
      Structured combined bundle. This can be promoted directly into
      FullSolutionBundle after validation.
    """

    schema_version: str = "questions_answers_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[QuestionsAnswersQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# 4. Full solution schema
# ---------------------------------------------------------------------


class FullSolutionPart(BaseModel):
    """
    Input:
      Created by merging:
        - QuestionsOnlyBundle + AnswersOnlyBundle
      or by promoting:
        - QuestionsAnswersBundle

    Output:
      One complete gradeable part used to generate reference_current.tex
      and reference_bundle.json.
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

    source_question_file: str = ""
    source_answer_file: str = ""


class FullSolutionQuestion(BaseModel):
    """
    Input:
      Complete gradeable question assembled from one or more uploads.

    Output:
      Stable full question object used by grading and TeX generation.
    """

    question_id: int
    parts: list[FullSolutionPart] = Field(default_factory=list)


class FullSolutionBundle(BaseModel):
    """
    Input:
      Final assembled solution-bank object.

      It can be created from:
        1. questions_only + answers_only
        2. questions_answers
        3. old ReferenceBundle-style data

    Output:
      The canonical full solution used to write:
        - full_solution_bundle.json
        - reference_bundle.json
        - reference_current.tex
        - reference_summary.json
    """

    schema_version: str = "full_solution_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[FullSolutionQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)

    def to_json_schema_for_openai(self) -> dict[str, Any]:
        return self.model_json_schema()


# ---------------------------------------------------------------------
# Backward compatibility with existing code
# ---------------------------------------------------------------------


class ReferencePart(FullSolutionPart):
    """
    Backward-compatible name.

    Input:
      Existing code that still expects ReferencePart.

    Output:
      Same shape as FullSolutionPart.
    """
    pass


class ReferenceQuestion(BaseModel):
    """
    Backward-compatible name.

    Input:
      Existing code that still expects ReferenceQuestion.

    Output:
      Same logical shape as FullSolutionQuestion, but using ReferencePart.
    """

    question_id: int
    parts: list[ReferencePart] = Field(default_factory=list)


class ReferenceBundle(BaseModel):
    """
    Backward-compatible name.

    Input:
      Existing code that still expects ReferenceBundle.

    Output:
      Same logical shape as FullSolutionBundle.

    Later we can remove this once all code uses FullSolutionBundle directly.
    """

    schema_version: str = "reference_bundle_v1"

    exam_id: str = ""
    exam_title: str = ""

    questions: list[ReferenceQuestion] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    structure_corrections: list[StructureCorrection] = Field(default_factory=list)

    source_names: list[str] = Field(default_factory=list)

    def to_json_schema_for_openai(self) -> dict[str, Any]:
        return self.model_json_schema()


def questions_only_bundle_schema() -> dict[str, Any]:
    return QuestionsOnlyBundle.model_json_schema()


def answers_only_bundle_schema() -> dict[str, Any]:
    return AnswersOnlyBundle.model_json_schema()


def questions_answers_bundle_schema() -> dict[str, Any]:
    return QuestionsAnswersBundle.model_json_schema()


def full_solution_bundle_schema() -> dict[str, Any]:
    return FullSolutionBundle.model_json_schema()


def reference_bundle_schema() -> dict[str, Any]:
    """
    Backward-compatible schema function.
    """
    return ReferenceBundle.model_json_schema()