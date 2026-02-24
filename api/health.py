# api/health.py
from __future__ import annotations

import os
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    ollama_model = os.getenv("OLLAMA_MODEL", "unknown")
    google_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    google_key_present = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    return {
        "status": "ok",
        "ollama_model": ollama_model,
        "google_model": google_model if google_key_present else None,
        "google_configured": google_key_present,
    }