from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from core.config import RUNS_ROOT, MATHPIX_APP_ID, MATHPIX_APP_KEY
from core.security import require_session
from grader.ocr.mathpix_client import MathpixError, mathpix_text_to_student_tex, process_image_or_pdf

router = APIRouter(prefix="/routes", tags=["ocr"])

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


@router.post("/ocr_handwritten")
async def ocr_handwritten(
    file: UploadFile = File(...),
    _session: dict = Depends(require_session),
) -> dict:
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        raise HTTPException(status_code=400, detail="Missing MATHPIX_APP_ID / MATHPIX_APP_KEY in environment.")

    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PNG, JPG, JPEG, WEBP, or PDF.",        )

    run_dir = Path(tempfile.mkdtemp(prefix="mathocr_", dir=str(RUNS_ROOT)))
    src_dir = run_dir / "ocr_input"
    src_dir.mkdir(parents=True, exist_ok=True)

    saved_path = src_dir / filename
    saved_path.write_bytes(await file.read())

    try:
        result = await run_in_threadpool(
            process_image_or_pdf,
            file_path=saved_path,
            app_id=MATHPIX_APP_ID,
            app_key=MATHPIX_APP_KEY,
            include_line_data=True,
        )
    except MathpixError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    # Save raw result for debugging / later review UI
    out_path = run_dir / "mathpix_raw.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    detected_text = result.get("text", "") or ""
    generated_tex = mathpix_text_to_student_tex(detected_text, source_name=filename)

    tex_path = run_dir / "ocr_student_answer.tex"
    tex_path.write_text(generated_tex, encoding="utf-8")

    return {
        "ok": True,
        "uploaded_filename": filename,
        "saved_path": str(saved_path),
        "raw_json_path": str(out_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": result.get("text", ""),
        "html": result.get("html", ""),
        "latex_styled": result.get("latex_styled", ""),
        "student_tex": generated_tex,
        "student_tex_path": str(tex_path),
        "mathpix_mode": result.get("_mathpix_mode"),
        "pdf_id": result.get("pdf_id"),
        "pdf_status": result.get("status"),
        "line_count": len(result.get("line_data", []) or []),
        "is_handwritten": result.get("is_handwritten"),
        "confidence": result.get("confidence"),
    }