import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from core.config import RUNS_ROOT
from core.security import require_session
from core.ai_clients.ocr_client import OcrClientError, run_ocr
from schemas.ocr_response import OcrOptions
from services.ocr_services.student_work_ocr import build_student_work_ocr_result

router = APIRouter(prefix="/routes", tags=["ocr"])

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


@router.post("/ocr_handwritten")
async def ocr_handwritten(
    file: UploadFile = File(...),
    ocr_provider: str = Form("mathpix"),
    ocr_model: str = Form(""),
    _session: dict = Depends(require_session),
) -> dict:

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
        ocr_result = await run_in_threadpool(
            run_ocr,
            file_path=saved_path,
            provider=ocr_provider,
            model=ocr_model or None,
            options=OcrOptions(
                temperature=0.0,
                max_output_tokens=12000,
                timeout_s=300,
                language_hint="hebrew",
                preserve_math=True,
                preserve_layout=True,
                include_line_data=True,
            ),
        )
    except OcrClientError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    # Save provider-neutral OCR result for debugging / later review UI
    out_path = run_dir / "ocr_response.json"
    out_path.write_text(
        json.dumps(ocr_result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    student_result = build_student_work_ocr_result(
        ocr=ocr_result,
        source_name=filename,
        out_dir=run_dir,
    )

    detected_text = student_result.raw_ocr_text
    generated_tex = student_result.student_tex
    tex_path = Path(student_result.student_tex_path)

    return {
        "ok": True,
        "uploaded_filename": filename,
        "saved_path": str(saved_path),
        "raw_json_path": str(out_path),

        "ocr_schema_version": ocr_result.schema_version,
        "ocr_provider": ocr_result.provider,
        "ocr_model": ocr_result.model,
        "ocr_input_kind": ocr_result.input_kind,
        "ocr_provider_mode": ocr_result.provider_mode,
        "ocr_provider_document_id": ocr_result.provider_document_id,
        "ocr_provider_status": ocr_result.provider_status,
        "ocr_response_id": ocr_result.response_id,
        "ocr_usage": ocr_result.usage.model_dump(),

        "text": detected_text,
        "html": ocr_result.html,
        "latex_styled": ocr_result.latex_styled,

        "student_ocr_schema_version": student_result.schema_version,
        "student_tex": generated_tex,
        "student_tex_path": str(tex_path),
        "student_ocr_warnings": student_result.warnings,
        "student_ocr_needs_teacher_review": student_result.needs_teacher_review,

        "line_count": len(ocr_result.pages[0].line_data) if ocr_result.pages else 0,
        "is_handwritten": ocr_result.is_handwritten,
        "confidence": ocr_result.confidence,
    }