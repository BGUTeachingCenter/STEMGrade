import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from core.config import (
    GEMINI_OCR_MAX_OUTPUT_TOKENS,
    GOOGLE_OCR_MAX_OUTPUT_TOKENS,
    OCR_MAX_OUTPUT_TOKENS,
    OPENAI_OCR_MAX_OUTPUT_TOKENS,
    RUNS_ROOT,
)
from core.security import require_session
from core.ai_clients.ocr_client import OcrClientError, run_ocr
from schemas.ocr_response import AiUsage, OcrPage, OcrResponse
from core.ai_clients.ai_usage_logger import bind_usage_context
from core.debug import create_debug_trace
from schemas.ocr_response import OcrOptions
from services.handwritten_ocr.student_work_ocr import build_student_work_ocr_result
from services.ocr_routing import decide_student_ocr_path
from common.exam_summary import write_student_summary, read_json_file
from services.student_work_store import save_ocr_student_work
from services.student_access import get_student_code_record

router = APIRouter(prefix="/routes", tags=["ocr"])

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


def _structured_ocr_output_token_budget(
    provider: str,
) -> int:
    """
    Select the configured structured-OCR output budget.

    All values are defined and validated in core.config.
    """
    normalized = str(
        provider or ""
    ).strip().lower()

    if normalized in {
        "google",
        "ai_studio",
        "aistudio",
    }:
        return GOOGLE_OCR_MAX_OUTPUT_TOKENS

    if normalized == "gemini":
        return GEMINI_OCR_MAX_OUTPUT_TOKENS

    if normalized in {
        "openai",
        "gpt",
        "chatgpt",
    }:
        return OPENAI_OCR_MAX_OUTPUT_TOKENS

    return OCR_MAX_OUTPUT_TOKENS


@router.post("/ocr_handwritten")
async def ocr_handwritten(
    file: UploadFile = File(...),
    ocr_provider: str = Form("mathpix"),
    ocr_model: str = Form(""),
    ocr_document_mode: str = Form("auto"),
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
    session_role = _session.get("role")
    teacher_id = ""
    student_code = ""

    voucher_id = ""

    if session_role == "teacher":
        teacher_id = (
                _session.get("teacher_id")
                or _session.get("sub")
                or ""
        ).strip()

    elif session_role == "student":
        student_code = (_session.get("sub") or "").strip()

        try:
            student_record = get_student_code_record(student_code) or {}

            teacher_id = str(
                student_record.get("teacher_id")
                or student_record.get("owner_teacher_id")
                or _session.get("teacher_id")
                or ""
            ).strip()

            voucher_id = str(
                student_record.get("voucher_id")
                or student_record.get("voucher_hash")
                or student_record.get("voucher_code")
                or ""
            ).strip()

        except Exception:
            teacher_id = str(
                _session.get("teacher_id")
                or ""
            ).strip()
            voucher_id = ""

    bind_usage_context(
        debug_run_id=trace.run_id,
        route="ocr_handwritten",
        provider=ocr_provider,
        session_role=session_role,
        teacher_id=teacher_id or None,
        student_code=student_code or None,
        voucher_id=voucher_id or None,
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

    route_decision = await run_in_threadpool(
        decide_student_ocr_path,
        saved_path,
        ocr_document_mode,
    )
    trace.save_json("ocr_route_decision.json", route_decision.__dict__, stage="ocr_route")
    trace.log(
        "ocr_route",
        "selected",
        requested_mode=route_decision.requested_mode,
        document_type=route_decision.document_type,
        ocr_path=route_decision.ocr_path,
        reason=route_decision.reason,
        native_text_char_count=route_decision.native_text_char_count,
    )

    try:
        if route_decision.ocr_path == "native_pdf_text_layer":
            ocr_result = OcrResponse(
                provider="tex",
                model="native_pdf_text_layer",
                input_kind="pdf",
                source_filename=filename,
                source_path=str(saved_path),
                text=route_decision.native_text,
                pages=[OcrPage(page_index=0, page_number=1, text=route_decision.native_text)],
                confidence=1.0,
                is_handwritten=False,
                provider_mode="native_pdf_text_layer",
                provider_status="ok",
                usage=AiUsage(),
            )
        else:
            prompt_variant = "student_answer_bundle" if (
                route_decision.document_type == "handwritten_scan"
                and (ocr_provider or "").strip().lower() in {"openai", "gpt", "chatgpt", "google", "gemini", "ai_studio", "aistudio"}
            ) else "default"

            ocr_result = await run_in_threadpool(
                run_ocr,
                file_path=saved_path,
                provider=ocr_provider,
                model=ocr_model or None,
                options=OcrOptions(
                    temperature=0.0,
                    max_output_tokens=(
                        _structured_ocr_output_token_budget(
                            ocr_provider
                        )
                        if prompt_variant == "student_answer_bundle"
                        else OCR_MAX_OUTPUT_TOKENS
                    ),
                    timeout_s=420 if prompt_variant == "student_answer_bundle" else 300,
                    language_hint="hebrew",
                    preserve_math=True,
                    preserve_layout=True,
                    include_line_data=True,
                    prompt_variant=prompt_variant,
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
        document_type=route_decision.document_type,
        ocr_path=route_decision.ocr_path,
    )
    trace.save_json("student_work_ocr_result.json", student_result.model_dump(), stage="student_tex")
    if student_result.student_answer_bundle:
        trace.save_json("student_answer_bundle.json", student_result.student_answer_bundle, stage="student_answer_bundle")

    student_summary_path: Path | None = None
    student_summary: dict | None = None

    if student_result.student_tex_path:
        student_tex_path = Path(student_result.student_tex_path)
        trace.save_file(student_tex_path, "ocr_student_answer.tex", stage="student_tex")

        try:
            summary_source_path = Path(student_result.student_answer_bundle_path) if student_result.student_answer_bundle_path else student_tex_path
            student_summary_path = write_student_summary(summary_source_path, out_dir=run_dir)
            student_summary = read_json_file(student_summary_path) or {}
            trace.save_file(student_summary_path, "student_summary.json", stage="student_summary")
            trace.log(
                "student_summary",
                "built",
                parse_ok=bool(student_summary.get("parse_ok")),
                qnums=student_summary.get("qnums") or [],
                part_count=student_summary.get("part_count"),
                keys_preview=student_summary.get("keys_preview") or [],
            )
        except Exception as e:
            trace.log("student_summary", "failed", status="warning", error=str(e)[:500])

    trace.log(
        "student_tex",
        "generated",
        warnings=student_result.warnings,
        needs_teacher_review=student_result.needs_teacher_review,
        document_type=student_result.document_type,
        ocr_path=student_result.ocr_path,
        detected_question_count=student_result.detected_question_count,
        detected_part_count=student_result.detected_part_count,
        has_student_answer_bundle=bool(student_result.student_answer_bundle),
    )

    detected_text = student_result.raw_ocr_text
    generated_tex = student_result.student_tex
    tex_path = Path(student_result.student_tex_path)

    stored_work: dict | None = None
    if session_role == "student" and student_code:
        try:
            stored_work = save_ocr_student_work(
                student_code=student_code,
                teacher_id=teacher_id,
                source_filename=filename,
                uploaded_bytes=uploaded_bytes,
                ocr_provider=ocr_result.provider or ocr_provider,
                ocr_model=ocr_result.model or ocr_model or "",
                document_type=route_decision.document_type,
                ocr_path=route_decision.ocr_path,
                route_reason=route_decision.reason,
                debug_trace_id=trace.run_id,
                debug_trace_dir=str(trace.path) if trace.enabled else "",
                ocr_response_path=out_path,
                student_work_result=student_result,
                student_summary_path=student_summary_path,
            )
            trace.log(
                "student_work_store",
                "saved",
                work_id=stored_work.get("work_id"),
                primary_filename=stored_work.get("primary_filename"),
            )
        except Exception as e:
            # Do not fail OCR just because archival storage failed.
            trace.log("student_work_store", "failed", status="warning", error=str(e)[:500])

    return {
        "ok": True,
        "debug_trace_id": trace.run_id,
        "debug_trace_dir": str(trace.path) if trace.enabled else None,
        "uploaded_filename": filename,
        "saved_path": str(saved_path),
        "raw_json_path": str(out_path),
        "stored_work": stored_work,

        "ocr_route_requested_mode": route_decision.requested_mode,
        "ocr_document_type": route_decision.document_type,
        "ocr_path": route_decision.ocr_path,
        "ocr_route_reason": route_decision.reason,
        "native_text_char_count": route_decision.native_text_char_count,

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
        "student_answer_bundle": student_result.student_answer_bundle,
        "student_answer_bundle_json": (json.dumps(student_result.student_answer_bundle, ensure_ascii=False, indent=2) if student_result.student_answer_bundle else ""),
        "student_answer_bundle_path": student_result.student_answer_bundle_path or None,
        "student_summary_path": str(student_summary_path) if student_summary_path else None,
        "student_summary": student_summary or {},
        "student_ocr_warnings": student_result.warnings,
        "student_ocr_needs_teacher_review": student_result.needs_teacher_review,

        "line_count": len(ocr_result.pages[0].line_data) if ocr_result.pages else 0,
        "is_handwritten": ocr_result.is_handwritten,
        "confidence": ocr_result.confidence,
    }