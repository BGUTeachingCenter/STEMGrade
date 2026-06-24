# core/config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from current working dir


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ---------------------------------------------------------------------
# Project-level AI model defaults
# ---------------------------------------------------------------------
# These are the defaults MathGrade will use when the values are not present
# in .env or the operating-system environment.
#
# We also push them into os.environ below with setdefault(...) so older code
# that still reads os.getenv(...) directly sees the same values.
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_OCR_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_OCR_MODEL = "gemini-3.5-flash"
DEFAULT_MATHGRADE_OLLAMA_PROOFREAD = "0"


# Make config.py the source of default env values.
os.environ.setdefault("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
os.environ.setdefault("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
os.environ.setdefault("OPENAI_OCR_MODEL", DEFAULT_OPENAI_OCR_MODEL)
os.environ.setdefault("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
os.environ.setdefault("GOOGLE_MODEL", DEFAULT_GEMINI_MODEL)
os.environ.setdefault("GEMINI_OCR_MODEL", DEFAULT_GEMINI_OCR_MODEL)
os.environ.setdefault("GOOGLE_OCR_MODEL", DEFAULT_GEMINI_OCR_MODEL)
os.environ.setdefault("MATHGRADE_OLLAMA_PROOFREAD", DEFAULT_MATHGRADE_OLLAMA_PROOFREAD)

# Mathpix API key
MATHPIX_APP_ID = (os.getenv("MATHPIX_APP_ID") or "").strip()
MATHPIX_APP_KEY = (os.getenv("MATHPIX_APP_KEY") or "").strip()


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Runs + debug dirs
RUNS_ROOT = Path(os.getenv("MATHGRADE_RUNS_DIR", PROJECT_ROOT / "runs"))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = Path(os.getenv("MATHGRADE_DEBUG_DIR", PROJECT_ROOT / "debug_logs"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Teacher auth
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "").strip()

# Session auth
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "28800"))  # 8h
# Cookies are Secure unless explicitly disabled for local HTTP dev
COOKIE_SECURE = env_bool("COOKIE_SECURE", "1")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax").lower()

# CORS: comma-separated origin allowlist. Empty => same-origin only.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Production mode hides internal error details from API responses
PRODUCTION = env_bool("PRODUCTION", "0")

# Bank config
BANK_ROOT = Path(os.getenv("MATHGRADE_SOLUTION_BANK_DIR", PROJECT_ROOT / "solution_bank"))
BANK_ROOT.mkdir(parents=True, exist_ok=True)

AUTO_REFERENCE = env_bool("MATHGRADE_AUTO_REFERENCE", "0")

# Ollama config
OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")

# Fixed font for PDF rendering
FIXED_FONT = os.getenv("MATHGRADE_FONT", "Arial")
ARIAL_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")