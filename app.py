# app.py
from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.auth import router as auth_router
from web.error_handlers import register_error_handlers
from web.health import router as health_router
from web.template_pages import router as web_router
from flows.student_grading.routes import router as grading_router
from flows.solution_bank.routes import router as bank_router
from web.progress import router as progress_router
from web.stats import router as stats_router
from flows.handwritten_ocr.routes import router as ocr_router
from core.config import ALLOWED_ORIGINS, PRODUCTION, PROJECT_ROOT
from core.security import SESSION_SECRET_CONFIGURED

logger = logging.getLogger("mathgrade")

app = FastAPI(title="MathGrade Bundle Generator", version="1.0")
register_error_handlers(app)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")


@app.on_event("startup")
async def _warn_on_insecure_config() -> None:
    if not SESSION_SECRET_CONFIGURED:
        logger.warning(
            "SESSION_SECRET is not set — sessions are signed with an ephemeral "
            "per-process secret. Set SESSION_SECRET in the environment for "
            "stable, multi-worker sessions."
        )
    if PRODUCTION and not ALLOWED_ORIGINS:
        logger.warning(
            "PRODUCTION=1 but ALLOWED_ORIGINS is empty. CORS will reject all "
            "cross-origin requests; if the frontend is on another origin, set "
            "ALLOWED_ORIGINS=https://your.site."
        )


# Cookies require a specific origin (not "*") plus allow_credentials=True. If
# ALLOWED_ORIGINS is empty we serve only same-origin (frontend served by this
# app), so CORS is effectively disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router)
app.include_router(web_router)
app.include_router(grading_router)
app.include_router(bank_router)
app.include_router(progress_router)
app.include_router(auth_router)
app.include_router(stats_router)
app.include_router(ocr_router)
