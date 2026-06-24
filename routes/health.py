# routes/health.py
from __future__ import annotations

import os
from fastapi import APIRouter

router = APIRouter(tags=["health"])


def _env_present(*names: str) -> bool:
    return any(bool((os.getenv(name) or "").strip()) for name in names)


@router.get("/health")
def health():
    ollama_model = os.getenv("OLLAMA_MODEL", "unknown")

    google_model = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "unknown"
    google_ocr_model = (
        os.getenv("GOOGLE_OCR_MODEL")
        or os.getenv("GEMINI_OCR_MODEL")
        or google_model
        or "unknown"
    )
    google_key_present = _env_present("GEMINI_API_KEY", "GOOGLE_API_KEY")

    gpt_model = os.getenv("OPENAI_MODEL", "unknown")
    gpt_ocr_model = os.getenv("OPENAI_OCR_MODEL") or gpt_model or "unknown"
    gpt_key_present = _env_present("OPENAI_API_KEY")

    mathpix_configured = _env_present("MATHPIX_APP_ID") and _env_present("MATHPIX_APP_KEY")

    return {
        "status": "ok",
        "ollama_model": ollama_model,
        "google_model": google_model,
        "google_ocr_model": google_ocr_model,
        "google_configured": google_key_present,
        "gpt_model": gpt_model,
        "gpt_ocr_model": gpt_ocr_model,
        "gpt_configured": gpt_key_present,
        "mathpix_configured": mathpix_configured,
        "ocr_models": {
            "openai": gpt_ocr_model if gpt_key_present else None,
            "google": google_ocr_model if google_key_present else None,
            "mathpix": "configured" if mathpix_configured else None,
        },
    }
