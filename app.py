# app.py
from __future__ import annotations

import traceback
from fastapi import FastAPI, Request
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



app = FastAPI(title="MathGrade Bundle Generator", version="1.0")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": tb, "path": str(request.url)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(web_router)
app.include_router(grading_router)
app.include_router(bank_router)
app.include_router(progress_router)
app.include_router(auth_router)
app.include_router(stats_router)
app.include_router(ocr_router)