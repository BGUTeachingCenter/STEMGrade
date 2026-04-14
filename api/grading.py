# api/grading.py
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from time import perf_counter

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse

# ✅ run blocking work off the event loop
from starlette.concurrency import run_in_threadpool

from core.config import RUNS_ROOT, FIXED_FONT, BANK_ROOT
from core.debug import write_debug_log

from grader.ai_grading.solution_bank_matcher import pick_reference_with_exam_id
from grader.file_handling.bundler import generate_bundle
from grader.ai_grading.grader_sources import grade_tex
from grader.file_handling.feedback_tex import build_feedback_tex  # TeX-first feedback
from grader.file_handling.unified_tex import build_unified_tex
from grader.file_handling.student_tex import parse_student_tex_answers
from grader.file_handling.compile_tex_to_pdf import compile_tex_to_pdf

from api.student_log import log_student_submission
from api.progress import init_job, push, done, fail

router = APIRouter(prefix="/api", tags=["grading"])

DEBUG = False


# -------------------------
# Small helpers
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
    """
    If p is a PDF -> return as pdf.
    If p is a TeX -> return as text/plain.
    """
    if p.suffix.lower() == ".pdf":
        return FileResponse(path=str(p), media_type="application/pdf", filename="graded_test.pdf")
    return FileResponse(path=str(p), media_type="text/plain; charset=utf-8", filename="graded_union.tex")


def _require_provider_env(provider: str) -> None:
    provider = (provider or "").strip().lower()

    if provider in ("google", "gemini", "google_ai_studio", "aistudio"):
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(
                status_code=400,
                detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.",
            )

    if provider in ("chatgpt", "openai", "gpt"):
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_key:
            raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment.")


def _debug_push(job_id: str | None, debug: bool, msg: str) -> None:
    if job_id and debug:
        push(job_id, msg)


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


def _extract_bundle_tex(bundle_outputs: Any, out_dir: Path) -> Path:
    bundle_tex = None

    if hasattr(bundle_outputs, "bundle_tex") and getattr(bundle_outputs, "bundle_tex"):
        bundle_tex = Path(getattr(bundle_outputs, "bundle_tex"))
    elif isinstance(bundle_outputs, (str, Path)) and str(bundle_outputs).endswith(".tex"):
        bundle_tex = Path(bundle_outputs)

    if not bundle_tex or not bundle_tex.exists():
        produced = [p.name for p in out_dir.glob("*")]
        raise RuntimeError(f"No bundle TeX produced. Files in out/: {produced}")

    return bundle_tex


def _choose_final_result_path(
    *,
    raw_result_path: Path,
    ai_dir: Path,
    normalized_provider: str,
) -> Path:
    """
    Prefer PDF if it exists.
    Otherwise fall back to TeX.
    """
    if raw_result_path.exists() and raw_result_path.suffix.lower() == ".pdf":
        return raw_result_path

    fallback_tex = ai_dir / f"graded_{normalized_provider}.tex"
    if fallback_tex.exists():
        return fallback_tex

    if raw_result_path.exists():
        return raw_result_path

    raise RuntimeError("No final output file was produced (neither PDF nor TeX).")


# -------------------------
# Main grading flow
# -------------------------


async def _grade_tex_flow(
    *,
    provider: str,
    student_tex: UploadFile,
    job_id: str | None,
    student_code: str | None,
    request: Request | None,
    debug: bool = False,
) -> FileResponse:
    """
    Unified grading flow with debug support.

    New final flow:
      1) save student upload
      2) fetch matching reference
      3) generate Q/A bundle TeX
      4) grade with AI
      5) build feedback TeX
      6) unify bundle + feedback into one final TeX
      7) try compiling final TeX to PDF
      8) if compile fails, return the TeX instead
    """
    _require_provider_env(provider)
    normalized_provider = _normalized_provider_name(provider)

    persistent_results_dir = RUNS_ROOT / "final_results"
    persistent_results_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir: Path | None = None

    try:
        if job_id:
            init_job(job_id)

        # =========================================================
        # Stage 1: Create temp run workspace
        # =========================================================
        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        _debug_push(job_id, debug, f"Created temp run dir: {tmp_dir}")
        _debug_push(job_id, debug, f"Normalized provider name: {normalized_provider}")

        # =========================================================
        # Stage 2: Save student upload and generate summary
        # =========================================================
        tex_path = tmp_dir / (student_tex.filename or "student.tex")
        tex_path.write_bytes(await student_tex.read())

        if job_id:
            push(job_id, "Saved student file")
        _debug_push(job_id, debug, f"Student TeX saved at: {tex_path}")

        try:
            summary_path = _write_student_summary(tex_path, out_dir=out_dir)
            if job_id:
                push(job_id, f"Generated student summary: {summary_path.name}")
            _debug_push(job_id, debug, f"Student summary path: {summary_path}")
        except Exception as e:
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # =========================================================
        # Stage 3: Match the submission to a reference in the bank
        # =========================================================
        if job_id:
            push(job_id, "Fetching reference from solution bank…")

        if not BANK_ROOT.exists():
            raise RuntimeError(f"Solution bank folder does not exist: {BANK_ROOT}")

        t_ref = perf_counter()
        exam_id, chosen_ref = await run_in_threadpool(
            pick_reference_with_exam_id,
            bank_dir=BANK_ROOT,
            student_tex=tex_path,
            prefer_heuristic=True,
            llm_top_k=12,
        )
        ref_secs = perf_counter() - t_ref

        if job_id:
            push(job_id, f"Fetched: {exam_id} (reference match {ref_secs:.1f}s)")
        _debug_push(job_id, debug, f"Chosen reference: {chosen_ref}")

        ref_path = tmp_dir / Path(chosen_ref).name
        ref_path.write_text(
            Path(chosen_ref).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
        _debug_push(job_id, debug, f"Copied reference to: {ref_path}")

        # =========================================================
        # Stage 4: Build the Q/A bundle first
        # =========================================================
        if job_id:
            push(job_id, "Generating Q/A bundle (TeX)…")

        t_bundle = perf_counter()
        bundle_outputs = await run_in_threadpool(
            generate_bundle,
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=out_dir,
            font_name=FIXED_FONT,
            compile_pdf=False,
        )
        bundle_secs = perf_counter() - t_bundle

        if job_id:
            push(job_id, f"Q/A bundle TeX ready ({bundle_secs:.1f}s)")

        bundle_tex = _extract_bundle_tex(bundle_outputs, out_dir)
        _debug_push(job_id, debug, f"Bundle TeX path: {bundle_tex}")

        # =========================================================
        # Stage 5: Grade with AI
        # =========================================================
        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)
        _debug_push(job_id, debug, f"AI output dir: {ai_dir}")

        if job_id:
            push(job_id, "Grading please wait…")

        t_grade = perf_counter()
        grades_json, _ = await run_in_threadpool(
            grade_tex,
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model=provider,
            debug=debug,
            log_fn=(lambda msg: push(job_id, msg)) if job_id else None,
        )
        grade_secs = perf_counter() - t_grade

        if job_id:
            push(job_id, f"Grading finished ({grade_secs:.1f}s)")
        _debug_push(job_id, debug, f"Grades JSON path: {grades_json}")

        # =========================================================
        # Stage 6: Read usage info and log submission
        # =========================================================
        gemini_tokens = 0
        try:
            usage_path = ai_dir / "usage_summary.json"
            if usage_path.exists():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
                if (usage.get("provider") or "").lower() in (
                    "google",
                    "gemini",
                    "google_ai_studio",
                    "aistudio",
                ):
                    gemini_tokens = int(usage.get("total_tokens") or 0)
                _debug_push(job_id, debug, f"Usage summary path: {usage_path}")
        except Exception:
            gemini_tokens = 0

        try:
            if student_code:
                ua = request.headers.get("user-agent", "") if request else ""
                ip = request.client.host if (request and request.client) else ""
                log_student_submission(
                    code=student_code,
                    exam_id=str(exam_id),
                    provider=normalized_provider,
                    gemini_tokens=gemini_tokens,
                    ip=ip,
                    user_agent=ua,
                )
        except Exception:
            pass

        # =========================================================
        # Stage 7: Build feedback TeX
        # =========================================================
        if job_id:
            push(job_id, "Generating feedback (TeX)…")

        t_fb = perf_counter()
        feedback_tex = await run_in_threadpool(
            build_feedback_tex,
            grades_json=grades_json,
            out_dir=ai_dir,
        )
        fb_secs = perf_counter() - t_fb

        if job_id:
            push(job_id, f"Feedback TeX ready ({fb_secs:.1f}s)")
        _debug_push(job_id, debug, f"Feedback TeX path: {feedback_tex}")

        # =========================================================
        # Stage 8: Unify bundle + feedback into one final TeX
        # =========================================================
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

        if job_id:
            push(job_id, f"Unified TeX ready ({union_secs:.1f}s)")
        _debug_push(job_id, debug, f"Unified TeX path: {final_tex}")

        # =========================================================
        # Stage 9: Final step only — try compiling unified TeX to PDF
        # =========================================================
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
                if job_id:
                    push(job_id, f"PDF build finished ({pdf_secs:.1f}s)")
                    push(job_id, "Final output is PDF")
            else:
                result_path = final_tex
                if job_id:
                    push(job_id, f"PDF build did not produce a valid PDF ({pdf_secs:.1f}s)")
                    push(job_id, "Falling back to TeX")
        except Exception as e:
            pdf_secs = perf_counter() - t_pdf
            result_path = final_tex
            _debug_push(job_id, debug, f"Final PDF compile failed: {_safe_str(e, 500)}")
            if job_id:
                push(job_id, f"PDF compile failed ({pdf_secs:.1f}s); falling back to TeX")

        _debug_push(job_id, debug, f"Chosen final result path: {result_path}")

        # =========================================================
        # Stage 10: Persist final output and return it
        # =========================================================
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

        if job_id:
            push(job_id, f"Done. Sending file: {final_persistent_path.name}")
            done(job_id)

        return _response_for_path(final_persistent_path)

    except HTTPException:
        if job_id:
            fail(job_id, "HTTPException")
        raise

    except Exception as e:
        log_path = write_debug_log(f"grade_tex_{normalized_provider}", e)
        if job_id:
            push(job_id, f"FAILED: {_safe_str(e, 400)}")
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")

    finally:
        # =========================================================
        # Stage 11: Cleanup temp run dir unless debug mode is on
        # =========================================================
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
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    student_code: str | None = Form(None),
    request: Request = None,
):
    return await _grade_tex_flow(
        provider="ollama",
        student_tex=student_tex,
        job_id=job_id,
        student_code=student_code,
        request=request,
        debug=DEBUG,
    )


@router.post("/grade_tex_google")
async def grade_tex_google(
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    student_code: str | None = Form(None),
    request: Request = None,
):
    return await _grade_tex_flow(
        provider="gemini",
        student_tex=student_tex,
        job_id=job_id,
        student_code=student_code,
        request=request,
        debug=DEBUG,
    )


@router.post("/grade_tex_chatgpt")
async def grade_tex_chatgpt(
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    student_code: str | None = Form(None),
    request: Request = None,
):
    return await _grade_tex_flow(
        provider="chatgpt",
        student_tex=student_tex,
        job_id=job_id,
        student_code=student_code,
        request=request,
        debug=DEBUG,
    )