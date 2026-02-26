# core/config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from current working dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Runs + debug dirs
RUNS_ROOT = Path(os.getenv("MATHGRADE_RUNS_DIR", PROJECT_ROOT / "runs"))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = Path(os.getenv("MATHGRADE_DEBUG_DIR", PROJECT_ROOT / "debug_logs"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Teacher auth
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "").strip()

# Bank config
BANK_ROOT = Path(os.getenv("MATHGRADE_SOLUTION_BANK_DIR", PROJECT_ROOT / "solution_bank"))
BANK_ROOT.mkdir(parents=True, exist_ok=True)

AUTO_REFERENCE = os.getenv("MATHGRADE_AUTO_REFERENCE", "0").lower() in {"1", "true", "yes"}

# Ollama config (single source of truth)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL

# Fixed font for PDF rendering
FIXED_FONT = os.getenv("MATHGRADE_FONT", "Arial")
ARIAL_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()