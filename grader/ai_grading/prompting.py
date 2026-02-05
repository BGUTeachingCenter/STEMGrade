from __future__ import annotations

from pathlib import Path


PROMPT_FILENAME = "grading_prompt_he_math.txt"


def load_grading_prompt(filename: str = PROMPT_FILENAME) -> str:
    """Load the grading prompt from a txt file located next to this module."""

    here = Path(__file__).resolve().parent
    prompt_path = here / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()
