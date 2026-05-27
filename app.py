# app.py
from __future__ import annotations

import logging
import secrets
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import router as auth_router
from api.health import router as health_router
from api.web_pages import router as web_router
from api.grading import router as grading_router
from api.bank import router as bank_router
from api.progress import router as progress_router
from api.stats import router as stats_router
from api.ocr import router as ocr_router
from core.config import ALLOWED_ORIGINS, PRODUCTION
from core.debug import write_debug_log
from core.security import SESSION_SECRET_CONFIGURED

logger = logging.getLogger("mathgrade")

app = FastAPI(title="MathGrade Bundle Generator", version="1.0")


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


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    # Preserve intentional HTTP errors (401/403/400/etc) — they're safe to surface.
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_id = secrets.token_hex(8)
    log_path = write_debug_log(f"unhandled_{error_id}", exc)
    logger.error(
        "Unhandled error %s on %s: %s\n%s",
        error_id,
        request.url,
        exc,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
    if PRODUCTION:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "error_id": error_id},
        )
    # Dev: keep the old verbose response so debugging stays easy.
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "error_id": error_id,
            "log_path": log_path,
            "path": str(request.url),
        },
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
