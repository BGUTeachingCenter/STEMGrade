# routes/grading_routes.py
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from routes.progress import done, fail, init_job, push
from routes.student_log_routes import log_student_submission
from core.security import get_session
from core.config import BANK_ROOT, FIXED_FONT, RUNS_ROOT
from core.storage import require_safe_exam_id, write_reference_summary
from core.debug import create_debug_trace, write_debug_log
from core.security import require_session
from services.student_grading.grading.grader_payloads import grade_payload_manifest
from services.student_grading.grading.payloads import PayloadItem, build_payloads
from services.student_grading.grading.solution_bank_matcher import pick_reference_with_match_info
from services.student_grading.bundler import _write_bundle_tex_inline_answers  # uses your existing bundler writer
from common.tex.compile_tex_to_pdf import compile_tex_to_pdf
from services.student_grading.feedback_tex import build_feedback_tex
from common.tex.student_tex import parse_student_tex_answers
from common.exam_summary import compare_summaries, read_json_file, write_student_summary as write_student_summary_file
from services.student_grading.unified_tex import build_unified_tex
from core.ai_clients.ai_usage_logger import bind_usage_context

router = APIRouter(prefix="/routes", tags=["grading"])

DEBUG = True


# -------------------------
# Small helpers (already in your style)
# -------------------------

def _safe_str(x: Any, limit: int = 200) -> str:
    s = str(x) if x is not None else ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _fallback_student_qnums(tex_path: Path) -> list[int]:
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    found = set()

    for m in re.finditer(r"\b(?:Question|Q)\s*([0-9]{1,3})\b", text, flags=re.IGNORECASE):
        found.add(int(m.group(1)))

    for m in re.finditer(r"\bשאלה\s*([0-9]{1,3})\b", text):
        found.add(int(m.group(1)))

    return sorted(found)


def _extract_student_summary(student_tex_path: Path, *, out_dir: Path) -> dict:
    summary: dict[str, Any] = {
        "qnums": [],
        "keys_preview": [],
        "preview_text": "",
        "parse_ok": False,
    }

    try:
        student_answers, _ranges = parse_student_tex_answers(student_tex_path, out_dir)
        if student_answers:
            qnums = sorted({q for (q, _part) in student_answers.keys()})
            keys = []
            preview_lines = []

            for (q, part), body in list(student_answers.items())[:14]:
                part_str = (part or "").strip()
                key = f"Q{q}{part_str}"
                keys.append(key)

                snip = _safe_str(body, limit=140)
                preview_lines.append(f"{key}: {snip if snip else '(empty)'}")

            summary.update(
                {
                    "qnums": qnums,
                    "keys_preview": keys[:60],
                    "preview_text": "\n".join(preview_lines)[:6000],
                    "parse_ok": True,
                }
            )
            return summary

    except Exception as e:
        summary["parse_error"] = _safe_str(e, 500)

    qnums = _fallback_student_qnums(student_tex_path)
    summary.update(
        {
            "qnums": qnums,
            "keys_preview": [f"Q{q}" for q in qnums][:60],
            "preview_text": "(fallback qnum detection)",
            "parse_ok": False,
        }
    )
    return summary


def _write_student_summary(student_tex_path: Path, *, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "filename": student_tex_path.name,
        **_extract_student_summary(student_tex_path, out_dir=out_dir),
    }
    p = out_dir / "student_summary.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _response_for_path(p: Path) -> FileResponse:
    if p.suffix.lower() == ".pdf":
        return FileResponse(path=str(p), media_type="application/pdf", filename="graded_test.pdf")
    return FileResponse(path=str(p), media_type="text/plain; charset=utf-8", filename="graded_union.tex")


def _require_provider_env(provider: str) -> None:
    p = (provider or "").strip().lower()
    if p in ("google", "gemini", "google_ai_studio", "aistudio"):
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")

    if p in ("chatgpt", "openai", "gpt"):
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_key:
            raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment.")


def _safe_name(s: str, fallback: str = "unknown") -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _normalized_provider_name(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("google", "gemini", "google_ai_studio", "aistudio"):
        return "gemini"
    if p in ("chatgpt", "openai", "gpt"):
        return "gpt"
    return "ollama"


def _build_persistent_result_path(
    *,
    persistent_root: Path,
    student_code: str | None,
    exam_id: str,
    provider: str,
    source_result_path: Path,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    student_part = _safe_name(student_code or "unknown_student")
    exam_part = _safe_name(exam_id or "unknown_exam")
    suffix = source_result_path.suffix or ".pdf"

    target_dir = persistent_root / student_part / exam_part
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{ts}_graded_{provider}{suffix}"
    return target_dir / filename


def _write_result_metadata(
    *,
    meta_path: Path,
    student_code: str | None,
    exam_id: str,
    provider: str,
    final_path: Path,
    debug: bool,
) -> None:
    meta = {
        "student_code": student_code,
        "exam_id": exam_id,
        "provider": provider,
        "debug": debug,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "final_file": final_path.name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# Main grading flow (single extraction, single source-of-truth payloads)
# -------------------------

async def _grade_tex_flow(
    *,
    provider: str,
    student_tex: UploadFile,
    job_id: str | None,
    session: dict,
    request: Request | None,
    selected_exam_id: str | None = None,
    debug: bool = False,
) -> FileResponse:
    # Trust only the session for identity; never the form field.
    student_code = session["sub"] if session.get("role") == "student" else None
    """
    Canonical flow (no duplicate extraction):

      1) Save student upload
      2) Match reference in solution bank
      3) Build payloads ONCE (full_solution_bundle.json + student snippets when available)
      4) Write Q/A bundle TeX from those payloads (display/export only)
      5) Grade each payload (AI) -> grades.json (+ usage_summary.json)
      6) Build feedback TeX from grades.json
      7) Unify bundle + feedback into one TeX
      8) Compile unified TeX to PDF (fallback to TeX)
      9) Persist final output and return it
    """
    _require_provider_env(provider)
    normalized_provider = _normalized_provider_name(provider)

    trace = create_debug_trace(
        "grading",
        provider=normalized_provider,
        job_id=job_id,
        student_filename=student_tex.filename,
        session_role=session.get("role") if session else None,
        student_code=student_code,
    )
    bind_usage_context(
        debug_run_id=trace.run_id,
        route="grade_tex",
        provider=normalized_provider,
        student_code=student_code,
    )
    trace.log("grading", "started", provider=normalized_provider, student_filename=student_tex.filename, selected_exam_id=selected_exam_id if session.get("role") == "teacher" else None)

    persistent_results_dir = RUNS_ROOT / "final_results"
    persistent_results_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir: Path | None = None
    student_summary_path: Path | None = None

    try:
        if job_id:
            init_job(job_id)

        # Stage 1: temp workspace
        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2: save student
        uploaded_name = Path(student_tex.filename or "student.tex").name
        tex_path = tmp_dir / uploaded_name
        student_upload_bytes = await student_tex.read()
        tex_path.write_bytes(student_upload_bytes)
        artifact_name = "student_answer_bundle.json" if tex_path.suffix.lower() == ".json" else f"student_upload{tex_path.suffix or '.tex'}"
        trace.save_bytes(artifact_name, student_upload_bytes, stage="student_upload")
        trace.log("student_upload", "saved", path=str(tex_path), bytes=len(student_upload_bytes), suffix=tex_path.suffix.lower())
        if job_id:
            push(job_id, "Saved student file")

        # summary (optional)
        try:
            summary_path = write_student_summary_file(tex_path, out_dir=out_dir)
            student_summary_path = summary_path
            trace.save_file(summary_path, "student_summary.json", stage="student_summary")
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                trace.log(
                    "student_summary",
                    "built",
                    parse_ok=bool(summary_data.get("parse_ok")),
                    qnums=summary_data.get("qnums") or [],
                    keys_preview=summary_data.get("keys_preview") or [],
                    preview_chars=len(summary_data.get("preview_text") or ""),
                )
            except Exception as summary_log_error:
                trace.log(
                    "student_summary",
                    "log_summary_failed",
                    status="warning",
                    error=_safe_str(summary_log_error, 300),
                )
            if job_id:
                push(job_id, f"Generated student summary: {summary_path.name}")
        except Exception as e:
            trace.log("student_summary", "failed", status="warning", error=_safe_str(e, 500))
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # Stage 3: match reference
        if job_id:
            push(job_id, "Fetching reference from solution bank…")

        if not BANK_ROOT.exists():
            raise RuntimeError(f"Solution bank folder does not exist: {BANK_ROOT}")

        t_ref = perf_counter()
        manual_exam_id = (selected_exam_id or "").strip() if session.get("role") == "teacher" else ""

        if manual_exam_id:
            exam_id = require_safe_exam_id(manual_exam_id)
            uploads = BANK_ROOT / exam_id / "uploads"
            json_ref = uploads / "full_solution_bundle.json"
            tex_ref = uploads / "reference_current.tex"
            if json_ref.exists():
                chosen_ref = json_ref
            elif tex_ref.exists():
                chosen_ref = tex_ref
            else:
                raise HTTPException(status_code=404, detail=f"Selected reference exam has no full_solution_bundle.json or reference_current.tex: {exam_id}")

            try:
                write_reference_summary(exam_id)
            except Exception as e:
                trace.log("reference_match", "manual_summary_refresh_failed", status="warning", error=_safe_str(e, 500))

            match_method = "manual_teacher_selection"
            match_reason = "Teacher test mode selected the reference explicitly."
            match_score = None
            student_qnums = []
            reference_qnums = []
            reference_summary_path = uploads / "reference_summary.json"
            try:
                if student_summary_path and student_summary_path.exists():
                    student_summary_for_log = read_json_file(student_summary_path) or {}
                    student_qnums = [int(x) for x in (student_summary_for_log.get("qnums") or []) if str(x).isdigit()]
                reference_summary_for_log = read_json_file(reference_summary_path) or {}
                reference_qnums = [int(x) for x in (reference_summary_for_log.get("qnums") or []) if str(x).isdigit()]
            except Exception:
                pass
        else:
            match = await run_in_threadpool(
                pick_reference_with_match_info,
                bank_dir=BANK_ROOT,
                student_tex=tex_path,
                prefer_heuristic=True,
                llm_top_k=12,
            )
            exam_id, chosen_ref = match.exam_id, match.path
            match_method = match.method
            match_reason = match.reason
            match_score = match.score
            student_qnums = list(match.student_qnums)
            reference_qnums = list(match.reference_qnums)

        ref_secs = perf_counter() - t_ref
        reference_source = "json" if Path(chosen_ref).suffix.lower() == ".json" else "tex_fallback"
        reference_selection_mode = "manual" if manual_exam_id else "automatic"
        trace.log(
            "reference_match",
            "selected",
            exam_id=str(exam_id),
            reference_path=str(chosen_ref),
            reference_source=reference_source,
            reference_selection_mode=reference_selection_mode,
            match_method=match_method,
            match_reason=match_reason,
            match_score=match_score,
            student_qnums=student_qnums,
            reference_qnums=reference_qnums,
            duration_s=round(ref_secs, 3),
        )
        if job_id:
            source_label = "JSON reference" if reference_source == "json" else "TeX fallback reference"
            mode_label = "manual" if manual_exam_id else "auto"
            push(job_id, f"Fetched: {exam_id} ({source_label}, {mode_label} reference, {ref_secs:.1f}s)")

        ref_path = tmp_dir / Path(chosen_ref).name
        ref_path.write_text(
            Path(chosen_ref).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
        matched_artifact_name = "matched_reference.json" if reference_source == "json" else "matched_reference.tex"
        trace.save_file(ref_path, matched_artifact_name, stage="reference_match")

        # Compare student_summary.json to reference_summary.json before building payloads.
        # This makes matching failures and OCR structure problems visible even when
        # we use the single-exam fallback.
        try:
            if student_summary_path is None or not student_summary_path.exists():
                student_summary_path = write_student_summary_file(tex_path, out_dir=out_dir)

            reference_summary_path = BANK_ROOT / str(exam_id) / "uploads" / "reference_summary.json"
            student_summary = read_json_file(student_summary_path) or {}
            reference_summary = read_json_file(reference_summary_path) or {}

            summary_compare = compare_summaries(
                student_summary=student_summary,
                reference_summary=reference_summary,
                exam_id=str(exam_id),
                reference_path=str(chosen_ref),
                match_method=match_method,
                match_reason=match_reason,
            )
            summary_compare_path = out_dir / "summary_compare.json"
            summary_compare_path.write_text(
                json.dumps(summary_compare, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            trace.save_file(summary_compare_path, "summary_compare.json", stage="reference_match")
            trace.log(
                "reference_match",
                "summary_compared",
                status=summary_compare.get("status", "ok"),
                qnum_overlap=summary_compare.get("overlap", {}).get("qnum_overlap", []),
                qnum_overlap_count=summary_compare.get("overlap", {}).get("qnum_overlap_count", 0),
                part_overlap_count=summary_compare.get("overlap", {}).get("part_overlap_count", 0),
                warnings=summary_compare.get("warnings", []),
            )
        except Exception as e:
            trace.log(
                "reference_match",
                "summary_compare_failed",
                status="warning",
                error=_safe_str(e, 500),
            )

        # Stage 4: build payloads ONCE (source-of-truth)
        if job_id:
            push(job_id, "Extracting parts (building payloads)…")

        payload_out = out_dir / "payloads_run"
        payload_out.mkdir(parents=True, exist_ok=True)

        t_payloads = perf_counter()
        manifest_json, items = await run_in_threadpool(
            build_payloads,
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=payload_out,
            default_max_points=0.0,
        )
        payload_secs = perf_counter() - t_payloads
        trace.save_file(manifest_json, "payload_manifest.json", stage="payloads")
        trace.log(
            "payloads",
            "built",
            duration_s=round(payload_secs, 3),
            payload_count=len(items),
            reference_source=reference_source,
        )
        for it in items[:80]:
            trace.save_file(it.payload_path, f"payloads/{it.payload_path.name}", stage="payloads")

        if job_id:
            push(job_id, f"Payloads ready ({payload_secs:.1f}s)")

        # Stage 4b: write bundle TeX from the SAME payloads (no re-extraction)
        if job_id:
            push(job_id, "Generating Q/A bundle (TeX)…")

        reference_snippets: Dict[Tuple[int, str | None], str] = {}
        student_answers: Dict[Tuple[int, str | None], str] = {}

        # We reuse the payload json content; keep keys aligned with bundler expectations.
        for it in items:
            data = json.loads(it.payload_path.read_text(encoding="utf-8"))
            key_dict = data.get("key") or {}
            qnum = int(key_dict.get("qnum"))
            part = key_dict.get("part") or ""
            key = (qnum, part)

            q_text = (data.get("reference") or {}).get("question_text") or ""
            sol_text = (data.get("reference") or {}).get("solution_text") or ""
            ref_block = "\n\n".join([x for x in [q_text.strip(), sol_text.strip()] if x.strip()]).strip()
            reference_snippets[key] = ref_block

            stu_raw = (data.get("student") or {}).get("latex_raw") or ""
            student_answers[key] = stu_raw.strip()

        bundle_tex = out_dir / "qa_bundle.tex"
        await run_in_threadpool(
            _write_bundle_tex_inline_answers,
            bundle_tex,
            title="Q/A Bundle (Reference + Student Answer)",
            font_name=FIXED_FONT,
            reference_snippets=reference_snippets,
            student_answers=student_answers,
        )
        trace.save_file(bundle_tex, "qa_bundle.tex", stage="qa_bundle")
        if job_id:
            push(job_id, "Q/A bundle TeX ready")

        # Stage 5: grade each payload (AI) using the SAME manifest
        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        if job_id:
            push(job_id, "Grading please wait…")

        t_grade = perf_counter()
        grades_json = await run_in_threadpool(
            grade_payload_manifest,
            manifest_json=manifest_json,
            out_dir=ai_dir,
            model=provider,
            debug=debug,
            log_fn=(lambda msg: push(job_id, msg)) if job_id else None,
        )
        grade_secs = perf_counter() - t_grade
        trace.save_file(grades_json, "grades.json", stage="grading")
        trace.log("grading", "completed", duration_s=round(grade_secs, 3), grades_path=str(grades_json))
        if job_id:
            push(job_id, f"Grading finished ({grade_secs:.1f}s)")

        # Stage 6: read usage and log submission (students only; admins are skipped in logger)
        gemini_tokens = 0
        try:
            usage_path = ai_dir / "usage_summary.json"
            if usage_path.exists():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
                trace.save_file(usage_path, "usage_summary.json", stage="usage")
                if (usage.get("provider") or "").lower() in ("google", "gemini", "google_ai_studio", "aistudio"):
                    gemini_tokens = int(usage.get("total_tokens") or 0)
        except Exception:
            gemini_tokens = 0

        try:
            session_role = session.get("role") if session else None

            # Only real student submissions are logged.
            # Teacher test-mode submissions are graded, but skipped here.
            if session_role == "student" and student_code:
                ua = request.headers.get("user-agent", "") if request else ""
                ip = request.client.host if (request and request.client) else ""

                log_student_submission(
                    code=student_code,
                    exam_id=str(exam_id),
                    provider=provider,
                    ip=ip,
                    user_agent=ua,
                    gemini_tokens=gemini_tokens,
                    session_role=session_role,
                )
        except Exception:
            pass

        # Stage 7: feedback TeX
        if job_id:
            push(job_id, "Generating feedback (TeX)…")

        t_fb = perf_counter()
        feedback_tex = await run_in_threadpool(
            build_feedback_tex,
            grades_json=grades_json,
            out_dir=ai_dir,
        )
        fb_secs = perf_counter() - t_fb
        trace.save_file(feedback_tex, "feedback.tex", stage="feedback_tex")
        trace.log("feedback_tex", "built", duration_s=round(fb_secs, 3), path=str(feedback_tex))
        if job_id:
            push(job_id, f"Feedback TeX ready ({fb_secs:.1f}s)")

        # Stage 8: unify bundle + feedback into one final TeX
        if job_id:
            push(job_id, "Combining Q/A bundle with feedback…")

        t_union = perf_counter()
        final_tex = await run_in_threadpool(
            build_unified_tex,
            qa_tex=bundle_tex,
            feedback_tex=feedback_tex,
            out_dir=ai_dir,
            output_stem=f"graded_{normalized_provider}",
        )
        union_secs = perf_counter() - t_union
        trace.save_file(final_tex, "final_unified.tex", stage="unified_tex")
        trace.log("unified_tex", "built", duration_s=round(union_secs, 3), path=str(final_tex))
        if job_id:
            push(job_id, f"Unified TeX ready ({union_secs:.1f}s)")

        # Stage 9: compile final TeX to PDF (fallback to TeX)
        if job_id:
            push(job_id, "Compiling final unified TeX to PDF…")

        result_path: Path
        t_pdf = perf_counter()
        try:
            compiled_pdf = await run_in_threadpool(
                lambda: compile_tex_to_pdf(
                    final_tex,
                    ai_dir,
                    clean=True,
                    font_name=FIXED_FONT,
                    passes=2,
                    texinputs=[final_tex.parent, bundle_tex.parent, feedback_tex.parent],
                ).pdf
            )
            compiled_pdf = Path(compiled_pdf)
            pdf_secs = perf_counter() - t_pdf

            if compiled_pdf.exists() and compiled_pdf.suffix.lower() == ".pdf":
                result_path = compiled_pdf
                trace.save_file(compiled_pdf, "final_output.pdf", stage="pdf_compile")
                trace.log("pdf_compile", "completed", duration_s=round(pdf_secs, 3), path=str(compiled_pdf))
                if job_id:
                    push(job_id, f"PDF build finished ({pdf_secs:.1f}s)")
            else:
                result_path = final_tex
                trace.log("pdf_compile", "missing_pdf", status="warning", duration_s=round(pdf_secs, 3))
                if job_id:
                    push(job_id, f"PDF build did not produce a valid PDF ({pdf_secs:.1f}s) — falling back to TeX")
        except Exception as e:
            pdf_secs = perf_counter() - t_pdf
            result_path = final_tex
            trace.error("pdf_compile", e, duration_s=round(pdf_secs, 3))
            if job_id:
                push(job_id, f"PDF compile failed ({pdf_secs:.1f}s); falling back to TeX: {_safe_str(e, 220)}")

        # Stage 10: persist final output
        final_persistent_path = _build_persistent_result_path(
            persistent_root=persistent_results_dir,
            student_code=student_code,
            exam_id=str(exam_id),
            provider=normalized_provider,
            source_result_path=result_path,
        )
        shutil.copy2(result_path, final_persistent_path)

        _write_result_metadata(
            meta_path=final_persistent_path.with_suffix(".json"),
            student_code=student_code,
            exam_id=str(exam_id),
            provider=normalized_provider,
            final_path=final_persistent_path,
            debug=debug,
        )
        trace.save_file(final_persistent_path, f"persisted_result{final_persistent_path.suffix}", stage="persist")
        trace.save_file(final_persistent_path.with_suffix(".json"), "persisted_result_metadata.json", stage="persist")
        trace.log(
            "grading",
            "finished",
            exam_id=str(exam_id),
            reference_source=reference_source,
            reference_selection_mode=reference_selection_mode,
            final_path=str(final_persistent_path),
        )

        if job_id:
            push(job_id, f"Done. Sending file: {final_persistent_path.name}")
            done(job_id)

        return _response_for_path(final_persistent_path)

    except HTTPException as e:
        trace.error("http_exception", e)
        if job_id:
            fail(job_id, "HTTPException")
        raise

    except Exception as e:
        trace.error("grading", e)
        log_path = write_debug_log(f"grade_tex_{_normalized_provider_name(provider)}", e)
        if job_id:
            push(job_id, f"FAILED: {_safe_str(e, 400)}")
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")

    finally:
        if tmp_dir and tmp_dir.exists():
            if debug:
                if job_id:
                    push(job_id, f"Debug mode ON: kept temp files in {tmp_dir}")
            else:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass


# -------------------------
# Endpoints
# -------------------------

@router.post("/grade_tex_ollama")
async def grade_tex_ollama(
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    session: dict = Depends(require_session),
):
    return await _grade_tex_flow(
        provider="ollama",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        selected_exam_id=selected_exam_id,
        debug=DEBUG,
    )


@router.post("/grade_tex_google")
async def grade_tex_google(
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    session: dict = Depends(require_session),
):
    # IMPORTANT: pass "google" (not "gemini") so client selection + schema pruning match
    return await _grade_tex_flow(
        provider="google",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        selected_exam_id=selected_exam_id,
        debug=DEBUG,
    )


@router.post("/grade_tex_chatgpt")
async def grade_tex_chatgpt(
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    session: dict = Depends(require_session),
):
    return await _grade_tex_flow(
        provider="chatgpt",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        debug=DEBUG,
    )