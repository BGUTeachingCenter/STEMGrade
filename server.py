from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from grader.qa_bundle import generate_qa_bundle_pdf

app = FastAPI(title="MathGrade Bundle Generator", version="1.0")

# Dev-friendly CORS. In production, set allow_origins to your real domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/api/generate")
async def generate_bundle(
    reference_pdf: UploadFile = File(...),
    student_tex: UploadFile = File(...),
):
    """
    Generate a QA bundle PDF from:
      - reference_pdf: official exam+solution PDF
      - student_tex: student's LaTeX results file

    Returns the final bundle PDF as a file download.

    Notes:
    - Uses a fixed font ("Arial") to avoid user-supplied font issues.
    - Uses a temporary folder per request (safe for concurrent users).
    - Cleans up the temporary folder AFTER streaming the response.
    """
    if not reference_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="reference_pdf must be a .pdf")

    if not student_tex.filename.lower().endswith((".tex", ".txt")):
        raise HTTPException(status_code=400, detail="student_tex must be .tex or .txt")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_"))
    out_dir = tmp_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploads
        ref_path = tmp_dir / "reference.pdf"
        tex_path = tmp_dir / "student.tex"

        with ref_path.open("wb") as f:
            shutil.copyfileobj(reference_pdf.file, f)

        with tex_path.open("wb") as f:
            shutil.copyfileobj(student_tex.file, f)

        # Run pipeline
        outputs = generate_qa_bundle_pdf(
            reference_pdf=ref_path,
            student_tex=tex_path,
            out_dir=out_dir,
            font_name="Arial",  # fixed to prevent user-caused font issues
        )

        # Try to locate produced PDF robustly
        candidate = None
        if hasattr(outputs, "bundle_pdf") and getattr(outputs, "bundle_pdf"):
            candidate = Path(getattr(outputs, "bundle_pdf"))
        elif hasattr(outputs, "pdf") and getattr(outputs, "pdf"):
            candidate = Path(getattr(outputs, "pdf"))
        elif isinstance(outputs, (str, Path)):
            candidate = Path(outputs)

        if not candidate or not candidate.exists():
            pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not pdfs:
                produced = [p.name for p in out_dir.glob("*")]
                raise RuntimeError(f"No PDF produced. Files in out/: {produced}")
            candidate = pdfs[0]

        # Cleanup temp folder AFTER response finishes streaming
        cleanup = BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)

        return FileResponse(
            path=str(candidate),
            media_type="application/pdf",
            filename="qa_bundle.pdf",
            background=cleanup,
        )

    except Exception as e:
        # If anything fails, clean up immediately and return a readable JSON error.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Generation failed", "detail": str(e)},
        )
