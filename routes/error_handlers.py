# routes/error_handlers.py
from __future__ import annotations

import logging
import secrets
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import PRODUCTION, PROJECT_ROOT
from core.debug import write_debug_log

logger = logging.getLogger("mathgrade")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


def _wants_json(request: Request) -> bool:
    """Keep API endpoints API-like, while browser page errors get HTML."""
    if request.url.path.startswith("/routes") or request.url.path == "/health":
        return True

    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


def _context(request: Request, **kwargs) -> dict:
    return {"request": request, **kwargs}


def register_error_handlers(app) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
        if _wants_json(request):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        template_name = f"errors/error_{exc.status_code}.html"
        if exc.status_code not in {403, 404, 405, 429, 500}:
            template_name = "errors/error_general.html"

        return templates.TemplateResponse(
            request=request,
            name=template_name,
            context=_context(
                request,
                status_code=exc.status_code,
                detail=exc.detail,
            ),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def custom_unhandled_exception_handler(request: Request, exc: Exception):
        error_id = secrets.token_hex(8)
        log_path = write_debug_log(f"unhandled_{error_id}", exc)
        logger.error(
            "Unhandled error %s on %s: %s\n%s",
            error_id,
            request.url,
            exc,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

        if _wants_json(request):
            if PRODUCTION:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "error_id": error_id},
                )

            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "error_id": error_id,
                    "log_path": log_path,
                    "path": str(request.url),
                },
            )

        detail = "Something went wrong."
        if not PRODUCTION:
            detail = f"{exc} · error_id={error_id}"

        return templates.TemplateResponse(
            request=request,
            name="errors/error_500.html",
            context=_context(
                request,
                status_code=500,
                detail=detail,
            ),
            status_code=500,
        )
