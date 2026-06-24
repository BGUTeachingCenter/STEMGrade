import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from core.config import RUNS_ROOT
from core.security import require_session
from core.ai_clients.ocr_client import OcrClientError, run_ocr
from core.ai_clients.ai_usage_logger import bind_usage_context
from core.debug import create_debug_trace
from schemas.ocr_response import OcrOptions
from services.handwritten_ocr.student_work_ocr import build_student_work_ocr_result

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

    trace = create_debug_trace(
        "student_ocr",
        provider=ocr_provider,
        model=ocr_model or None,
        source_name=filename,
    )
    bind_usage_context(
        debug_run_id=trace.run_id,
        route="ocr_handwritten",
        provider=ocr_provider,
    )
    trace.log("ocr", "started", filename=filename, suffix=suffix, provider=ocr_provider, model=ocr_model or None)

    run_dir = Path(tempfile.mkdtemp(prefix="mathocr_", dir=str(RUNS_ROOT)))
    src_dir = run_dir / "ocr_input"
    src_dir.mkdir(parents=True, exist_ok=True)

    saved_path = src_dir / filename
    uploaded_bytes = await file.read()
    saved_path.write_bytes(uploaded_bytes)
    trace.save_bytes(f"input/{filename}", uploaded_bytes, stage="upload")
    trace.log("upload", "saved", path=str(saved_path), bytes=len(uploaded_bytes))

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
        trace.error("ocr", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        trace.error("ocr", e)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    # Save provider-neutral OCR result for debugging / later review UI
    out_path = run_dir / "ocr_response.json"
    out_path.write_text(
        json.dumps(ocr_result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace.save_file(out_path, "ocr_response.json", stage="ocr")
    trace.save_text("ocr_primary_text.txt", ocr_result.primary_text(), stage="ocr")
    trace.log(
        "ocr",
        "completed",
        provider=ocr_result.provider,
        model=ocr_result.model,
        confidence=ocr_result.confidence,
        is_handwritten=ocr_result.is_handwritten,
        line_count=len(ocr_result.pages[0].line_data) if ocr_result.pages else 0,
    )

    student_result = build_student_work_ocr_result(
        ocr=ocr_result,
        source_name=filename,
        out_dir=run_dir,
    )
    trace.save_json("student_work_ocr_result.json", student_result.model_dump(), stage="student_tex")
    if student_result.student_tex_path:
        trace.save_file(student_result.student_tex_path, "ocr_student_answer.tex", stage="student_tex")
    trace.log(
        "student_tex",
        "generated",
        warnings=student_result.warnings,
        needs_teacher_review=student_result.needs_teacher_review,
    )

    detected_text = student_result.raw_ocr_text
    generated_tex = student_result.student_tex
    tex_path = Path(student_result.student_tex_path)

    return {
        "ok": True,
        "debug_trace_id": trace.run_id,
        "debug_trace_dir": str(trace.path) if trace.enabled else None,
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