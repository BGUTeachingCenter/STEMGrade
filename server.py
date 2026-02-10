from __future__ import annotations

import os
import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from grader.qa_bundle import generate_qa_bundle_pdf, generate_qa_bundle_from_reference_tex

# AI grading modules (you added these earlier)
from grader.ai_grading.grader import grade_bundle_pdf
from grader.ai_grading.grader_sources import grade_reference_and_student_tex, grade_reference_tex_and_student_tex
from grader.ai_grading.graded_pdf import build_graded_pdf
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import tempfile

RUNS_DIR = Path.cwd() / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# tmp_root = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_DIR)))


RUNS_ROOT = Path(r"C:\Users\alinag\PycharmProjects\MathTest\runs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = Path.cwd() / "debug_logs"
DEBUG_DIR.mkdir(exist_ok=True)

def write_debug_log(prefix: str, exc: Exception) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = DEBUG_DIR / f"{prefix}_{ts}.log"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    p.write_text(tb, encoding="utf-8")
    return str(p)



app = FastAPI(title="MathGrade Bundle Generator", version="1.0")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Show it in the response (great for local dev)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "traceback": tb,
            "path": str(request.url),
        },
    )

# Dev-friendly CORS. In production, set allow_origins to your real domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""MathGrade API server.

Single source of truth for Ollama configuration
----------------------------------------------
For vigorous model testing, change **only** `OLLAMA_MODEL` below.

All grading code reads the model from the environment (set here) and/or the
explicit parameter passed from this server.
"""

# Ollama config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Change THIS default to switch models for the whole app.
# You can still override it via environment variable if you prefer.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Propagate to the rest of the process so any module can read it consistently.
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL

# Fixed font to avoid user-supplied surprises
FIXED_FONT = os.getenv("MATHGRADE_FONT", "Arial")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "ollama_base_url": OLLAMA_BASE_URL, "ollama_model": OLLAMA_MODEL}


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Save an uploaded file to disk."""
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _pick_pdf_output(outputs) -> Optional[Path]:
    """
    Try to interpret the return value of generate_qa_bundle_pdf.
    Supports returning a dataclass with .bundle_pdf/.pdf or a direct path.
    """
    candidate = None
    if hasattr(outputs, "bundle_pdf") and getattr(outputs, "bundle_pdf"):
        candidate = Path(getattr(outputs, "bundle_pdf"))
    elif hasattr(outputs, "pdf") and getattr(outputs, "pdf"):
        candidate = Path(getattr(outputs, "pdf"))
    elif isinstance(outputs, (str, Path)):
        candidate = Path(outputs)
    return candidate


def _find_newest_pdf(folder: Path) -> Optional[Path]:
    """Return the newest PDF in folder, or None if none exist."""
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


async def _prepare_inputs(reference_pdf: UploadFile, student_tex: UploadFile) -> tuple[Path, Path, Path]:
    """
    Create temp dir and save uploads.
    Returns (tmp_dir, reference_pdf_path, student_tex_path).
    """
    if not reference_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="reference_pdf must be a .pdf")
    if not student_tex.filename.lower().endswith((".tex", ".txt")):
        raise HTTPException(status_code=400, detail="student_tex must be .tex or .txt")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_"))
    ref_path = tmp_dir / "reference.pdf"
    tex_path = tmp_dir / "student.tex"

    _save_upload(reference_pdf, ref_path)
    _save_upload(student_tex, tex_path)

    return tmp_dir, ref_path, tex_path


# def _file_response_with_cleanup(path: Path, download_name: str, tmp_dir: Path) -> FileResponse:
#     """Return a FileResponse and delete tmp_dir after streaming completes."""
#     cleanup = BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)
#     return FileResponse(
#         path=str(path),
#         media_type="application/pdf",
#         filename=download_name,
#         background=cleanup,
#     )

def _file_response_with_cleanup(path: Path, download_name: str, tmp_dir: Path) -> FileResponse:
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=download_name,
    )

# # New route for tex upload
# @app.post("/api/generate_tex")
# async def generate_bundle_from_reference_tex_api(
#     reference_tex: UploadFile = File(...),
#     student_tex: UploadFile = File(...),
# ):
#     """
#     Generate the bundle PDF (questions + official solutions + student answers)
#     using reference .tex (exam+solution) + student .tex.
#
#     Output: qa_bundle.pdf download.
#     """
#     tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_"))
#     try:
#         tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_"))
#         out_dir = tmp_dir / "out"
#         out_dir.mkdir(parents=True, exist_ok=True)
#
#         ref_path = tmp_dir / reference_tex.filename
#         tex_path = tmp_dir / student_tex.filename
#         ref_path.write_bytes(await reference_tex.read())
#         tex_path.write_bytes(await student_tex.read())
#
#         outputs = generate_qa_bundle_from_reference_tex(
#             reference_tex=ref_path,
#             student_tex=tex_path,
#             out_dir=out_dir,
#             font_name=FIXED_FONT,
#         )
#
#         bundle_pdf = _pick_pdf_output(outputs) or _find_newest_pdf(out_dir)
#         if not bundle_pdf or not bundle_pdf.exists():
#             produced = [p.name for p in out_dir.glob("*")]
#             raise RuntimeError(f"No bundle PDF produced. Files in out/: {produced}")
#
#         return _file_response_with_cleanup(bundle_pdf, "qa_bundle.pdf", tmp_dir)
#
#     except Exception as e:
#         log_path = write_debug_log("generate_tex", e)
#         raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")


@app.post("/api/grade_tex")
async def grade_bundle_from_reference_tex_api(
    reference_tex: UploadFile = File(...),
    student_tex: UploadFile = File(...),
):
    """Grade using reference .tex (questions/solutions) + student .tex.

    This avoids PDF extraction issues for math/RTL content.

    Output: graded_test.pdf (feedback + bundle).
    """
    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_path = tmp_dir / reference_tex.filename
        tex_path = tmp_dir / student_tex.filename
        ref_path.write_bytes(await reference_tex.read())
        tex_path.write_bytes(await student_tex.read())

        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        grades_json, _student_answers_unused = grade_reference_tex_and_student_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            ollama_base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
        )

        outputs = generate_qa_bundle_from_reference_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=out_dir,
            font_name=FIXED_FONT,
        )

        bundle_pdf = _pick_pdf_output(outputs) or _find_newest_pdf(out_dir)
        if not bundle_pdf or not bundle_pdf.exists():
            produced = [p.name for p in out_dir.glob("*")]
            raise RuntimeError(f"No bundle PDF produced. Files in out/: {produced}")

        font_path = Path(r"C:\\Windows\\Fonts\\arial.ttf")
        graded_pdf = build_graded_pdf(
            bundle_pdf=bundle_pdf,
            grades_json=grades_json,
            out_dir=ai_dir,
            font_path=font_path,
        )

        if not graded_pdf.exists():
            raise RuntimeError("Graded PDF was not created.")

        return _file_response_with_cleanup(graded_pdf, "graded_test.pdf", tmp_dir)

    except Exception as e:
        log_path = write_debug_log("grade_tex", e)
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")
        return JSONResponse(status_code=500, content={"error": "Grading failed", "detail": str(e)})
