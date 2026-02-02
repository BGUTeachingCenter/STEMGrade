# grader/config.py
from __future__ import annotations

from pathlib import Path

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

INPUT_TEX = Path("012.tex")
BUILD_DIR = Path("build")

# Must produce build/012_clean.pdf
COMPILE_SCRIPT = "compile_tex.py"

CLEAN_PDF = BUILD_DIR / "012_clean.pdf"
GRADE_JSON = BUILD_DIR / "grade.json"
PROMPTS_TXT = BUILD_DIR / "prompts_used.txt"

RAW_MODEL_TXT = BUILD_DIR / "raw_model_output.txt"
RAW_MODEL_EXTRACTED = BUILD_DIR / "raw_model_extracted.json"

GRADED_TEX = BUILD_DIR / "graded_report.tex"
GRADED_PDF = BUILD_DIR / "graded_report.pdf"

OLLAMA_TIMEOUT = 900
