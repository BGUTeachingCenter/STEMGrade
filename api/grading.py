# api/grading.py
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from core.config import RUNS_ROOT, FIXED_FONT, ARIAL_FONT_PATH, AUTO_REFERENCE, BANK_ROOT
from core.debug import write_debug_log

from grader import pick_reference_from_bank
from grader.qa_bundle import generate_qa_bundle_from_reference_tex
from grader.ai_grading.grader_sources import grade_reference_tex_and_student_tex
from grader.ai_grading.graded_pdf import build_graded_pdf

router = APIRouter(prefix="/api", tags=["grading"])

def _pick_pdf_output(outputs) -> Path | None:
    if hasattr(outputs, "bundle_pdf") and getattr(outputs, "bundle_pdf"):
        return Path(getattr(outputs, "bundle_pdf"))
    if hasattr(outputs, "pdf") and getattr(outputs, "pdf"):
        return Path(getattr(outputs, "pdf"))
    if isinstance(outputs, (str, Path)):
        return Path(outputs)
    return None

def _find_newest_pdf(folder: Path) -> Path | None:
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None

def _file_response(path: Path, download_name: str) -> FileResponse:
    # NOTE: You previously skipped cleanup. Keeping same behavior (no surprises).
    return FileResponse(path=str(path), media_type="application/pdf", filename=download_name)

@router.post("/grade_tex_ollama")
async def grade_tex_ollama(
    reference_tex: UploadFile = File(...),
    student_tex: UploadFile = File(...),
):
    """Grade using reference .tex + student .tex with Ollama.

    If AUTO_REFERENCE is enabled (env MATHGRADE_AUTO_REFERENCE=1) OR reference filename is AUTO,
    the server picks the best reference.tex from the solution bank and ignores the uploaded reference.
    """
    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save student first (for matching)
        tex_path = tmp_dir / (student_tex.filename or "student.tex")
        tex_path.write_bytes(await student_tex.read())

        use_bank = AUTO_REFERENCE or ((reference_tex.filename or "").lower() in {"auto", "autodetect", "bank"})

        if use_bank:
            chosen_ref = pick_reference_from_bank(
                bank_dir=BANK_ROOT,
                student_tex=tex_path,
                prefer_heuristic=True,
                llm_top_k=12,
            )
            ref_path = tmp_dir / Path(chosen_ref).name
            ref_path.write_text(Path(chosen_ref).read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        else:
            ref_path = tmp_dir / (reference_tex.filename or "reference.tex")
            ref_path.write_bytes(await reference_tex.read())

        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        grades_json, _ = grade_reference_tex_and_student_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model="ollama",
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

        graded_pdf = build_graded_pdf(
            bundle_pdf=bundle_pdf,
            grades_json=grades_json,
            out_dir=ai_dir,
            font_path=ARIAL_FONT_PATH,
        )
        if not graded_pdf.exists():
            raise RuntimeError("Graded PDF was not created.")

        return _file_response(graded_pdf, "graded_test.pdf")

    except Exception as e:
        log_path = write_debug_log("grade_tex_ollama", e)
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")

@router.post("/grade_tex_google")
async def grade_tex_google(
    reference_tex: UploadFile = File(...),
    student_tex: UploadFile = File(...),
):
    """Grade using reference .tex + student .tex with Google (Gemini).

    Requires GOOGLE_API_KEY (or GEMINI_API_KEY).
    """
    tmp_dir = None
    try:
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")

        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_path = tmp_dir / (reference_tex.filename or "reference.tex")
        tex_path = tmp_dir / (student_tex.filename or "student.tex")
        ref_path.write_bytes(await reference_tex.read())
        tex_path.write_bytes(await student_tex.read())

        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        grades_json, _ = grade_reference_tex_and_student_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model="google",
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

        graded_pdf = build_graded_pdf(
            bundle_pdf=bundle_pdf,
            grades_json=grades_json,
            out_dir=ai_dir,
            font_path=ARIAL_FONT_PATH,
        )
        if not graded_pdf.exists():
            raise RuntimeError("Graded PDF was not created.")

        return _file_response(graded_pdf, "graded_test.pdf")

    except HTTPException:
        raise
    except Exception as e:
        log_path = write_debug_log("grade_tex_google", e)
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")