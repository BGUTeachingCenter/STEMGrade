from __future__ import annotations

from pathlib import Path

from services.teacher_profiles import SUBJECT_LABELS, normalize_subject

PROMPT_FILENAME = "grading_prompt_he_math.txt"


def _subject_prompt_filename(subject: str | None) -> str:
    return f"grading_prompt_he_{normalize_subject(subject)}.txt"


def load_grading_prompt(
    filename: str = PROMPT_FILENAME,
    *,
    subject: str | None = None,
    extra_instructions: str = "",
) -> str:
    """Load the grading prompt located next to this module.

    If subject is provided, first try grading_prompt_he_<subject>.txt.
    Missing subject-specific prompts fall back to the math prompt, with a
    short subject note appended. This keeps old math grading stable while
    allowing gradual STEM expansion.
    """

    here = Path(__file__).resolve().parent
    requested_filename = _subject_prompt_filename(subject) if subject else filename
    prompt_path = here / requested_filename

    if not prompt_path.exists():
        prompt_path = here / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    normalized_subject = normalize_subject(subject)
    subject_label = SUBJECT_LABELS.get(normalized_subject, normalized_subject)

    if subject and requested_filename != filename and prompt_path.name == filename:
        prompt += (
            "\n\n"
            "Subject context:\n"
            f"This is a {subject_label} assessment. Apply the same careful grading logic, "
            "but prioritize the conventions, units, notation, code, diagrams, and reasoning patterns of this subject."
        )

    extra = (extra_instructions or "").strip()
    if extra:
        prompt += "\n\nTeacher/solution-specific grading instructions:\n" + extra

    return prompt
