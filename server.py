from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from grader.qa_bundle import generate_qa_bundle_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/generate")
async def generate_bundle(
    reference_pdf: UploadFile = File(...),
    student_tex: UploadFile = File(...),
    font_name: str = Form("Arial"),
):
    if not reference_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="reference_pdf must be a .pdf")
    if not student_tex.filename.lower().endswith((".tex", ".txt")):
        raise HTTPException(status_code=400, detail="student_tex must be .tex or .txt")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_"))
    try:
        ref_path = tmp_dir / "reference.pdf"
        tex_path = tmp_dir / "student.tex"
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save uploads
        with ref_path.open("wb") as f:
            shutil.copyfileobj(reference_pdf.file, f)
        with tex_path.open("wb") as f:
            shutil.copyfileobj(student_tex.file, f)

        # Run pipeline
        outputs = generate_qa_bundle_pdf(
            reference_pdf=ref_path,
            student_tex=tex_path,
            out_dir=out_dir,
            font_name=font_name,
        )

        # Try to locate produced PDF robustly
        candidate = None
        if hasattr(outputs, "bundle_pdf") and outputs.bundle_pdf:
            candidate = Path(outputs.bundle_pdf)
        elif hasattr(outputs, "pdf") and outputs.pdf:
            candidate = Path(outputs.pdf)
        elif isinstance(outputs, (str, Path)):
            candidate = Path(outputs)

        if not candidate or not candidate.exists():
            pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not pdfs:
                produced = [p.name for p in out_dir.glob("*")]
                raise RuntimeError(f"No PDF produced. Files in out/: {produced}")
            candidate = pdfs[0]

        # IMPORTANT: delete temp folder AFTER response finishes streaming
        cleanup = BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)

        return FileResponse(
            path=str(candidate),
            media_type="application/pdf",
            filename=candidate.name,
            background=cleanup,
        )

    except Exception as e:
        # If something fails, clean up immediately
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=500, content={"error": "Generation failed", "detail": str(e)})
