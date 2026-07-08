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
# App branding / teacher-facing vocabulary
# ---------------------------------------------------------------------
# Keep these as display settings first. Do not mass-rename internal folders,
# routes, env vars, or JSON files until the STEM generalization is stable.
APP_NAME = env_str("APP_NAME", "STEMGrade")
APP_LOGO_MARK = env_str("APP_LOGO_MARK", "⚛")
APP_DOMAIN_LABEL = env_str("APP_DOMAIN_LABEL", "STEM")
APP_HOME_SUBTITLE = env_str("APP_HOME_SUBTITLE", "AI-assisted STEM grading and feedback")
APP_SUBTITLE = env_str(
    "APP_SUBTITLE",
    "AI-assisted STEM grading and reference-bank management",
)
REFERENCE_BANK_LABEL = env_str("REFERENCE_BANK_LABEL", "Reference Bank")
ASSESSMENT_LABEL = env_str("ASSESSMENT_LABEL", "Assessment")
DEFAULT_SUBJECT = env_str("DEFAULT_SUBJECT", "math")

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

# Canonical course storage root:
#   data/teachers/<teacher_id>/courses/<voucher_id>/
#
# Each course/voucher will contain:
#   - solution_bank/
#   - students/
#   - course_manifest.json
#   - upload_log.json
TEACHERS_ROOT = Path(
    os.getenv("MATHGRADE_TEACHERS_ROOT", PROJECT_ROOT / "data" / "teachers")
)
TEACHERS_ROOT.mkdir(parents=True, exist_ok=True)

# Runs + debug dirs
RUNS_ROOT = Path(os.getenv("MATHGRADE_RUNS_DIR", PROJECT_ROOT / "runs"))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = Path(os.getenv("MATHGRADE_DEBUG_DIR", PROJECT_ROOT / "debug_logs"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Admin / legacy auth
# ADMIN_PASSWORD is the new app-owner/admin password. TEACHER_PASSWORD remains
# as a backward-compatible legacy teacher/admin code for local deployments that
# have not migrated to teacher profiles yet.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

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

# Teacher-specific solution banks.
# Each teacher profile gets a separate bank under:
#   data/teacher_banks/<teacher_id>/
TEACHER_BANK_ROOT = Path(
    os.getenv("MATHGRADE_TEACHER_BANK_DIR", PROJECT_ROOT / "data" / "teacher_banks")
)
TEACHER_BANK_ROOT.mkdir(parents=True, exist_ok=True)

# Teacher profiles/vouchers are intentionally stored outside the solution bank.
# This keeps access control data separate from teacher-uploaded reference files.
TEACHER_DATA_ROOT = Path(
    os.getenv("MATHGRADE_TEACHER_DATA_DIR", PROJECT_ROOT / "data" / "teacher_profiles")
)
TEACHER_DATA_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECT_OPTIONS = [
    "math",
    "physics",
    "chemistry",
    "biology",
    "cs",
    "engineering",
    "general_stem",
]
SUBJECT_LABELS = {
    "math": "Math",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "biology": "Biology",
    "cs": "Computer Science",
    "engineering": "Engineering",
    "general_stem": "General STEM",
}

AUTO_REFERENCE = env_bool("MATHGRADE_AUTO_REFERENCE", "0")

# Ollama config
OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")

# Fixed font for PDF rendering
FIXED_FONT = os.getenv("MATHGRADE_FONT", "Arial")
ARIAL_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")