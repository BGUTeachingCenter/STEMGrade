# api/grading.py
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse

from core.config import RUNS_ROOT, FIXED_FONT, ARIAL_FONT_PATH, BANK_ROOT
from core.debug import write_debug_log

from grader import pick_reference_from_bank
from grader.qa_bundle import generate_qa_bundle_from_reference_tex
from grader.ai_grading.grader_sources import grade_reference_tex_and_student_tex
from grader.ai_grading.graded_pdf import build_graded_pdf

from grader.student_tex import parse_student_tex_answers  # ✅ for student summary

from api.progress import init_job, push, done, fail  # ✅ progress

router = APIRouter(prefix="/api", tags=["grading"])


# -------------------------
# Helpers
# -------------------------

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


def _safe_str(x: Any, limit: int = 200) -> str:
    s = str(x) if x is not None else ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _fallback_student_qnums(tex_path: Path) -> list[int]:
    """
    Cheap fallback if parse_student_tex_answers fails.
    Looks for patterns like 'Question 3', 'Q3', 'שאלה 3'.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    found = set()

    # English patterns
    for m in re.finditer(r"\b(?:Question|Q)\s*([0-9]{1,3})\b", text, flags=re.IGNORECASE):
        found.add(int(m.group(1)))

    # Hebrew pattern: שאלה 3
    for m in re.finditer(r"\bשאלה\s*([0-9]{1,3})\b", text):
        found.add(int(m.group(1)))

    return sorted(found)


def _extract_student_summary(student_tex_path: Path, *, out_dir: Path) -> dict:
    """
    Create a summary similar in spirit to the reference summary:
      - qnums
      - keys like Q1a, Q2b (if parsable)
      - preview lines of the first few answers (short)
    """
    summary: dict[str, Any] = {
        "qnums": [],
        "keys_preview": [],
        "preview_text": "",
        "parse_ok": False,
    }

    try:
        student_answers, _ranges = parse_student_tex_answers(student_tex_path, out_dir)
        if student_answers:
            # keys are typically (qnum, part)
            qnums = sorted({q for (q, _part) in student_answers.keys()})
            keys = []
            preview_lines = []

            # build up to N preview entries
            for (q, part), body in list(student_answers.items())[:14]:
                part_str = (part or "").strip()
                key = f"Q{q}{part_str}"
                keys.append(key)

                # short preview from answer content
                snip = _safe_str(body, limit=140)
                if snip:
                    preview_lines.append(f"{key}: {snip}")
                else:
                    preview_lines.append(f"{key}: (empty)")

            summary.update(
                {
                    "qnums": qnums,
                    "keys_preview": keys[:60],
                    "preview_text": "\n".join(preview_lines)[:6000],
                    "parse_ok": True,
                }
            )
            return summary

        # If parsed but empty, fall through to fallback
    except Exception as e:
        summary["parse_error"] = _safe_str(e, 500)

    # Fallback: regex qnums only
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
    """
    Generate + write summary JSON for this student submission.
    Safe to call multiple times (overwrites summary).

    Writes: out_dir/student_summary.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "filename": student_tex_path.name,
        **_extract_student_summary(student_tex_path, out_dir=out_dir),
    }

    p = out_dir / "student_summary.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -------------------------
# Endpoints
# -------------------------

@router.post("/grade_tex_ollama")
async def grade_tex_ollama(
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
):
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

        # Student summary (new)
        try:
            p = write_student_summary(tex_path, out_dir=out_dir)
            if job_id:
                push(job_id, f"Generated student summary: {p.name}")
        except Exception as e:
            # Don't block grading if summary fails
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # Pick reference
        if job_id:
            push(job_id, "Fetching reference from solution bank…")

        if not BANK_ROOT.exists():
            raise RuntimeError(f"Solution bank folder does not exist: {BANK_ROOT}")

        chosen_ref = pick_reference_from_bank(
            bank_dir=BANK_ROOT,
            student_tex=tex_path,
            prefer_heuristic=True,
            llm_top_k=12,
        )

        if job_id:
            push(job_id, f"Fetched: {Path(chosen_ref).name}")

        # Copy reference into run folder
        ref_path = tmp_dir / Path(chosen_ref).name
        ref_path.write_text(
            Path(chosen_ref).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        # Grade
        if job_id:
            push(job_id, "Grading please wait…")

        grades_json, _ = grade_reference_tex_and_student_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model="ollama",
        )

        # Bundle
        if job_id:
            push(job_id, "Generating feedback bundle…")

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

        # Build graded PDF
        if job_id:
            push(job_id, "Building graded PDF…")

        graded_pdf = build_graded_pdf(
            bundle_pdf=bundle_pdf,
            grades_json=grades_json,
            out_dir=ai_dir,
            font_path=ARIAL_FONT_PATH,
        )
        if not graded_pdf.exists():
            raise RuntimeError("Graded PDF was not created.")

        if job_id:
            push(job_id, "Done. Sending PDF…")
            done(job_id)

        return _file_response(graded_pdf, "graded_test.pdf")

    except Exception as e:
        log_path = write_debug_log("grade_tex_ollama", e)
        if job_id:
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")


@router.post("/grade_tex_google")
async def grade_tex_google(
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
):
    tmp_dir: Path | None = None
    try:
        if job_id:
            init_job(job_id)

        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")

        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save student
        tex_path = tmp_dir / (student_tex.filename or "student.tex")
        tex_path.write_bytes(await student_tex.read())
        if job_id:
            push(job_id, "Saved student file")

        # Student summary (new)
        try:
            p = write_student_summary(tex_path, out_dir=out_dir)
            if job_id:
                push(job_id, f"Generated student summary: {p.name}")
        except Exception as e:
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # Pick reference
        if job_id:
            push(job_id, "Fetching reference from solution bank…")

        if not BANK_ROOT.exists():
            raise RuntimeError(f"Solution bank folder does not exist: {BANK_ROOT}")

        chosen_ref = pick_reference_from_bank(
            bank_dir=BANK_ROOT,
            student_tex=tex_path,
            prefer_heuristic=True,
            llm_top_k=12,
        )

        if job_id:
            push(job_id, f"Fetched: {Path(chosen_ref).name}")

        # Copy reference into run folder
        ref_path = tmp_dir / Path(chosen_ref).name
        ref_path.write_text(
            Path(chosen_ref).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        # Grade
        if job_id:
            push(job_id, "Grading please wait…")

        grades_json, _ = grade_reference_tex_and_student_tex(
            reference_tex=ref_path,
            student_tex=tex_path,
            out_dir=ai_dir,
            model="google",
        )

        # Bundle
        if job_id:
            push(job_id, "Generating feedback bundle…")

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

        # Build graded PDF
        if job_id:
            push(job_id, "Building graded PDF…")

        graded_pdf = build_graded_pdf(
            bundle_pdf=bundle_pdf,
            grades_json=grades_json,
            out_dir=ai_dir,
            font_path=ARIAL_FONT_PATH,
        )
        if not graded_pdf.exists():
            raise RuntimeError("Graded PDF was not created.")

        if job_id:
            push(job_id, "Done. Sending PDF…")
            done(job_id)

        return _file_response(graded_pdf, "graded_test.pdf")

    except HTTPException:
        if job_id:
            fail(job_id, "HTTPException")
        raise
    except Exception as e:
        log_path = write_debug_log("grade_tex_google", e)
        if job_id:
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")