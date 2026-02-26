# api/grading.py
from __future__ import annotations

import json
import os
import re
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
from grader.file_handling.qa_bundle import generate_qa_bundle_from_reference_tex
from grader.ai_grading.grader_sources import grade_reference_tex_and_student_tex
from grader.file_handling.feedback_tex import build_feedback_tex  # TeX-first feedback
from grader.file_handling.graded_pdf import build_graded_pdf      # unions 2 tex and compiles once
from grader.file_handling.student_tex import parse_student_tex_answers

from api.student_log import log_student_submission
from api.progress import init_job, push, done, fail

router = APIRouter(prefix="/api", tags=["grading"])


# -------------------------
# Helpers
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


def write_student_summary(student_tex_path: Path, *, out_dir: Path) -> Path:
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
    suf = p.suffix.lower()
    if suf == ".pdf":
        return FileResponse(path=str(p), media_type="application/pdf", filename="graded_test.pdf")
    return FileResponse(path=str(p), media_type="text/plain; charset=utf-8", filename="graded_union.tex")


def _require_provider_env(provider: str) -> None:
    if provider == "google":
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")
    if provider == "chatgpt":
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_key:
            raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment.")


async def _grade_tex_flow(
    *,
    provider: str,
    student_tex: UploadFile,
    job_id: str | None,
    student_code: str | None,
    request: Request | None,
) -> FileResponse:
    """
    Unified grading flow.
    Timing is measured around the TRUE heavy steps:
      - pick reference
      - grade (LLM calls)
      - build Q/A bundle TeX
      - build feedback TeX
      - compile union TeX (XeLaTeX)
    """
    _require_provider_env(provider)

    tmp_dir: Path | None = None
    try:
        if job_id:
            init_job(job_id)

        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save student
        tex_path = tmp_dir / (student_tex.filename or "student.tex")
        tex_path.write_bytes(await student_tex.read())
        if job_id:
            push(job_id, "Saved student file")

        # Student summary
        try:
            p = write_student_summary(tex_path, out_dir=out_dir)
            if job_id:
                push(job_id, f"Generated student summary: {p.name}")
        except Exception as e:
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # Pick reference (timed)
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

        # Copy reference into run folder
        ref_path = tmp_dir / Path(chosen_ref).name
        ref_path.write_text(
            Path(chosen_ref).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

        # Work dirs
        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        # Grade (LLM calls) (timed)
        if job_id:
            push(job_id, "Grading please wait…")

        t_grade = perf_counter()
        grades_json, _ = await run_in_threadpool(
            grade_reference_tex_and_student_tex,
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model=provider,
        )
        grade_secs = perf_counter() - t_grade
        if job_id:
            push(job_id, f"Grading finished ({grade_secs:.1f}s)")

        # Read Gemini token usage summary (if produced)
        gemini_tokens = 0
        try:
            usage_path = ai_dir / "usage_summary.json"
            if usage_path.exists():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
                if (usage.get("provider") or "").lower() in ("google", "gemini", "google_ai_studio", "aistudio"):
                    gemini_tokens = int(usage.get("total_tokens") or 0)
        except Exception:
            gemini_tokens = 0

        # ✅ Log ONCE (students only; teacher codes omitted by log_student_submission)
        try:
            if student_code:
                ua = request.headers.get("user-agent", "") if request else ""
                ip = request.client.host if (request and request.client) else ""
                log_student_submission(
                    code=student_code,
                    exam_id=str(exam_id),
                    provider=provider,
                    gemini_tokens=gemini_tokens,
                    ip=ip,
                    user_agent=ua,
                )
        except Exception:
            pass

        # Generate bundle TeX (timed)
        if job_id:
            push(job_id, "Generating Q/A bundle (TeX)…")

        t_bundle = perf_counter()
        bundle_outputs = await run_in_threadpool(
            generate_qa_bundle_from_reference_tex,
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=out_dir,
            font_name=FIXED_FONT,
            # compile_pdf=False,  # if supported in your function signature
        )
        bundle_secs = perf_counter() - t_bundle
        if job_id:
            push(job_id, f"Q/A bundle TeX ready ({bundle_secs:.1f}s)")

        # Locate bundle_tex
        bundle_tex = None
        if hasattr(bundle_outputs, "bundle_tex") and getattr(bundle_outputs, "bundle_tex"):
            bundle_tex = Path(getattr(bundle_outputs, "bundle_tex"))
        elif isinstance(bundle_outputs, (str, Path)) and str(bundle_outputs).endswith(".tex"):
            bundle_tex = Path(bundle_outputs)

        if not bundle_tex or not bundle_tex.exists():
            produced = [p.name for p in out_dir.glob("*")]
            raise RuntimeError(f"No bundle TeX produced. Files in out/: {produced}")

        # Build feedback TeX (timed)
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

        # Compile union TeX (timed)
        if job_id:
            push(job_id, "Building final output (compile union TeX)…")

        t_pdf = perf_counter()
        result_path = await run_in_threadpool(
            build_graded_pdf,
            qa_tex=bundle_tex,
            feedback_tex=feedback_tex,
            out_dir=ai_dir,
            font_name=FIXED_FONT,
        )
        pdf_secs = perf_counter() - t_pdf
        if job_id:
            push(job_id, f"PDF build finished ({pdf_secs:.1f}s)")
            push(job_id, "Done. Sending file…")
            done(job_id)

        return _response_for_path(result_path)

    except HTTPException:
        if job_id:
            fail(job_id, "HTTPException")
        raise
    except Exception as e:
        log_path = write_debug_log(f"grade_tex_{provider}", e)
        if job_id:
            push(job_id, f"FAILED: {_safe_str(e, 400)}")
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")


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
    )


@router.post("/grade_tex_google")
async def grade_tex_google(
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    student_code: str | None = Form(None),
    request: Request = None,
):
    return await _grade_tex_flow(
        provider="google",
        student_tex=student_tex,
        job_id=job_id,
        student_code=student_code,
        request=request,
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
    )