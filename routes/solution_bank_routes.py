# routes/solution_bank_routes.py
from __future__ import annotations

from pathlib import Path
import json
import re
from datetime import datetime
import shutil
from pydantic import BaseModel
from typing import Optional


from core.storage import exam_dir  # add this import

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from core.security import require_teacher
from core.storage import require_safe_exam_id, require_safe_filename, uploads_dir, write_reference_summary
from core.config import BANK_ROOT
from grader.file_handling.reference_tex import parse_reference_tex

from core.config import MATHPIX_APP_ID, MATHPIX_APP_KEY
from grader.ocr.mathpix_client import MathpixError, process_image_or_pdf
from grader.ai_grading.reference_builder import (
    build_questions_bundle_from_mathpix,
    build_reference_bundle_from_mathpix,
    bundle_to_exam_structure,
    questions_bundle_to_tex,
    write_reference_bundle_json,
    write_questions_bundle_json,
)

from routes.progress import init_job, push, done, fail


router = APIRouter(prefix="/routes/bank", tags=["bank"])

#-----------
# Helpers
#-----------


def _progress(job_id: str | None, msg: str) -> None:
    if not job_id:
        return
    try:
        push(job_id, msg)
    except Exception:
        # progress should never break upload
        pass


def _cleanup_empty_exam_folder(exam_id: str) -> None:
    """
    Remove empty solution-bank folders after the last reference file is deleted.

    Expected structure:
      BANK_ROOT / exam_id / uploads / reference_current.tex
    """
    d = exam_dir(exam_id)
    uploads = uploads_dir(exam_id)

    # If there are still TeX files anywhere under this exam, keep the folder.
    if d.exists() and any(d.rglob("*.tex")):
        return

    # Remove empty uploads folder first.
    try:
        if uploads.exists() and not any(uploads.iterdir()):
            uploads.rmdir()
    except Exception:
        pass

    # Remove known generated summary if it is now orphaned.
    for orphan_name in ("reference_summary.json",):
        orphan = d / orphan_name
        try:
            if orphan.exists():
                orphan.unlink()
        except Exception:
            pass

    # Remove exam folder only if it is empty.
    try:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    except Exception:
        pass


class ExamRenameReq(BaseModel):
    old_exam_id: str
    new_exam_id: str


class ExamDeleteReq(BaseModel):
    exam_id: str


TEX_SUFFIXES = {".tex", ".txt"}
OCR_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
ALL_BANK_UPLOAD_SUFFIXES = TEX_SUFFIXES | OCR_SUFFIXES


def _safe_bank_upload_filename(name: str) -> str:
    """
    Safe filename for teacher uploads into the solution bank.

    Do NOT use require_safe_filename here because that helper is currently
    TeX-specific in this project and rejects PDFs/images.
    """
    safe = Path(name or "upload.bin").name.strip()

    if not safe:
        raise HTTPException(status_code=400, detail="Empty filename")

    if safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise HTTPException(status_code=400, detail="Unsafe filename")

    suffix = Path(safe).suffix.lower()
    if suffix not in ALL_BANK_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .tex, .txt, .pdf, .png, .jpg, .jpeg, or .webp.",
        )

    return safe


async def _bank_upload_to_tex(
    *,
    upload: UploadFile,
    exam_id: str,
    content_type: str,
) -> tuple[bytes, dict]:
    """
    Convert either a TeX upload or an OCR-able upload into TeX bytes.

    Returns:
      (tex_bytes, extra_meta)
    """
    original_name = upload.filename or "upload.bin"
    safe_original_name = _safe_bank_upload_filename(original_name)
    suffix = Path(safe_original_name).suffix.lower()

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    if suffix not in ALL_BANK_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .tex, .txt, .pdf, .png, .jpg, .jpeg, or .webp.",
        )

    if suffix in TEX_SUFFIXES:
        return raw, {
            "source_kind": "tex",
            "original_filename": safe_original_name,
            "ocr_used": False,
            "pdf_text_layer_used": False,
        }

    uploads = uploads_dir(exam_id)

    # PDFs intentionally go through Mathpix.
    # Do not use embedded PDF text-layer extraction here because it destroys
    # Hebrew spacing and math structure in many exams.

    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        raise HTTPException(
            status_code=400,
            detail="Missing MATHPIX_APP_ID / MATHPIX_APP_KEY in environment.",
        )

    originals_dir = uploads / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    original_path = originals_dir / safe_original_name
    original_path.write_bytes(raw)

    try:
        result = process_image_or_pdf(
            file_path=original_path,
            app_id=MATHPIX_APP_ID,
            app_key=MATHPIX_APP_KEY,
            include_line_data=True,
        )
    except MathpixError as e:
        raise HTTPException(status_code=502, detail=f"Mathpix OCR failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    ocr_raw_path = uploads / f"{content_type}_mathpix_raw.json"
    ocr_raw_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ocr_text = result.get("text", "") or ""

    mathpix_text_path = uploads / f"{content_type}_mathpix_text.{result.get('downloaded_ext') or 'txt'}"
    mathpix_text_path.write_text(ocr_text, encoding="utf-8")

    # Remove Mathpix markdown image links. They are useful for debugging,
    # but not valid inside the temporary TeX.
    ocr_text_for_tex = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", ocr_text)

    tex = "\n".join(
        [
            r"\documentclass[12pt]{article}",
            r"\usepackage[a4paper,margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,mathtools}",
            r"\usepackage{fontspec}",
            r"\usepackage{polyglossia}",
            r"\setmainlanguage{hebrew}",
            r"\setotherlanguage{english}",
            r"\newfontfamily\hebrewfont{Arial}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.6em}",
            "",
            r"\begin{document}",
            rf"% OCR generated for solution bank. Source: {safe_original_name}",
            "",
            ocr_text_for_tex.strip(),
            "",
            r"\end{document}",
            "",
        ]
    )

    return tex.encode("utf-8"), {
        "source_kind": "mathpix",
        "original_filename": safe_original_name,
        "ocr_used": True,
        "ocr_raw_path": str(ocr_raw_path),
        "mathpix_text_path": str(mathpix_text_path),
        "mathpix_text": ocr_text,
        "mathpix_mode": result.get("_mathpix_mode"),
        "pdf_id": result.get("pdf_id"),
        "downloaded_ext": result.get("downloaded_ext"),
    }


def _strip_tex_wrapper_for_preview(tex_text: str) -> str:
    """
    Show useful body text for teacher preview without forcing reference-parser format.
    """
    text = (tex_text or "").replace("\r\n", "\n").replace("\r", "\n")

    begin = text.find(r"\begin{document}")
    end = text.find(r"\end{document}")

    if begin >= 0:
        text = text[begin + len(r"\begin{document}"):]
    if end >= 0:
        text = text[:end]

    # Remove comments and excessive empty lines for preview readability.
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("%"):
            continue
        lines.append(line)

    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _preview_from_exam_structure(structure: dict | None, tex_text: str) -> tuple[list[str], str]:
    """
    Preview questions-only files using the extracted exam structure,
    not parse_reference_tex.
    """
    structure = structure or {}
    questions = structure.get("questions") or []

    keys: list[str] = []
    preview_lines: list[str] = []

    if questions:
        preview_lines.append("Detected exam structure:")
        preview_lines.append("")

        for q in questions:
            qid = str(q.get("question_id", "")).strip()
            if not qid:
                continue

            parts = [str(p).strip() for p in (q.get("parts") or []) if str(p).strip()]

            if parts:
                keys.extend([f"Q{qid}{p}" for p in parts])
                preview_lines.append(f"Question {qid}: parts {', '.join(parts)}")
            else:
                keys.append(f"Q{qid}")
                preview_lines.append(f"Question {qid}: no parts detected")

        preview_lines.append("")
        preview_lines.append("Extracted text preview:")
        preview_lines.append("")

    else:
        preview_lines.append("No structure detected yet.")
        preview_lines.append("")
        preview_lines.append("Extracted text preview:")
        preview_lines.append("")

    body_preview = _strip_tex_wrapper_for_preview(tex_text)
    preview_lines.append(body_preview[:3500])

    return keys[:80], "\n".join(preview_lines).strip()


#-----------
# Routes
#-----------

@router.post("/upload")
async def upload_to_bank(
    exam_id: str = Form(...),
    content_type: str = Form(...),
    tex_file: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
    _session: dict = Depends(require_teacher),
):
    """Upload a teacher TeX file into the solution bank.

    - content_type: "reference" or "questions_only"
    - saves file as reference_current.tex / questions_only_current.tex
    - writes a lightweight .meta.json for fast preview/list
    """
    exam_id = require_safe_exam_id(exam_id)
    if content_type not in {"reference", "questions_only"}:
        raise HTTPException(status_code=400, detail="content_type must be reference or questions_only")

    if job_id:
        init_job(job_id)

    _progress(job_id, f"Starting upload for {content_type}: {tex_file.filename}")

    _progress(job_id, "Extracting file with Mathpix if needed...")

    raw, extra_meta = await _bank_upload_to_tex(
        upload=tex_file,
        exam_id=exam_id,
        content_type=content_type,
    )

    _progress(
        job_id,
        f"Extraction complete. Source: {extra_meta.get('source_kind')}, mode: {extra_meta.get('mathpix_mode') or 'tex'}",
    )

    uploads = uploads_dir(exam_id)

    suffix = "reference" if content_type == "reference" else "questions_only"
    filename = f"{suffix}_current.tex"
    tex_path = uploads / filename
    tex_text_for_structure = raw.decode("utf-8", errors="replace")

    structure = None
    structure_path = None

    if content_type == "questions_only":
        try:
            mathpix_text = extra_meta.get("mathpix_text") or tex_text_for_structure

            _progress(job_id, "AI is organizing the exam into question/part JSON...")

            questions_bundle = build_questions_bundle_from_mathpix(
                mathpix_text=mathpix_text,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                exam_id=exam_id,
            )

            bundle_path = uploads / "questions_only_bundle.json"
            write_questions_bundle_json(questions_bundle, bundle_path)

            _progress(job_id, "Saved questions_only_bundle.json")

            structure = bundle_to_exam_structure(questions_bundle)
            structure_path = uploads / "exam_structure.json"
            structure_path.write_text(
                json.dumps(structure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            raw = questions_bundle_to_tex(questions_bundle).encode("utf-8")

            _progress(job_id, "Generated canonical questions_only_current.tex")

        except Exception as e:
            structure = None
            structure_path = None

            fail_path = uploads / "questions_only_ai_builder_error.txt"
            fail_path.write_text(str(e), encoding="utf-8")
            fail(job_id, f"Questions-only AI builder failed: {e}") if job_id else None

            raise HTTPException(
                status_code=500,
                detail=f"Questions-only AI builder failed. See {fail_path.name}.",
            )


    elif content_type == "reference":

        questions_bundle_path = uploads / "questions_only_bundle.json"

        _progress(job_id, "Preparing to align official solution to questions bundle...")

        if not questions_bundle_path.exists():
            raise HTTPException(

                status_code=400,
                detail=(
                    "Upload the exam/questions-only file first. "
                    "reference upload requires questions_only_bundle.json."
                ),
            )

        try:
            questions_bundle = json.loads(
                questions_bundle_path.read_text(encoding="utf-8")
            )

            solution_mathpix_text = extra_meta.get("mathpix_text") or tex_text_for_structure

            _progress(job_id, "AI is aligning official solutions to the exam structure...")

            reference_bundle = build_reference_bundle_from_mathpix(
                questions_bundle=questions_bundle,
                solution_mathpix_text=solution_mathpix_text,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                exam_id=exam_id,
            )

            reference_bundle_path = uploads / "reference_bundle.json"

            write_reference_bundle_json(reference_bundle, reference_bundle_path)

            _progress(job_id, "Saved reference_bundle.json")

            corrections = reference_bundle.get("structure_corrections") or []
            high_conf_corrections = [
                c for c in corrections
                if str(c.get("confidence", "")).lower() in {"high", "גבוה", "strong"}
            ]

            if high_conf_corrections:
                corrected_questions_path = uploads / "questions_only_bundle_corrected_by_reference.json"
                write_questions_bundle_json(reference_bundle, corrected_questions_path)

                corrected_structure = bundle_to_exam_structure(reference_bundle)
                corrected_structure_path = uploads / "exam_structure_corrected_by_reference.json"
                corrected_structure_path.write_text(
                    json.dumps(corrected_structure, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                _progress(
                    job_id,
                    f"Saved corrected structure from reference upload ({len(high_conf_corrections)} high-confidence corrections).",
                )

            raw = questions_bundle_to_tex(reference_bundle).encode("utf-8")

            _progress(job_id, "Generated canonical reference_current.tex")


        except Exception as e:
            fail_path = uploads / "reference_ai_builder_error.txt"
            fail_path.write_text(str(e), encoding="utf-8")
            fail(job_id, f"Reference AI builder failed: {e}") if job_id else None
            raise HTTPException(
                status_code=500,
                detail=f"Reference AI builder failed. See {fail_path.name}.",
            )

    tex_path.write_bytes(raw)
    tex_text_for_structure = raw.decode("utf-8", errors="replace")

    # create/update reference_summary.json for fast matching
    if content_type == "reference":
        try:
            write_reference_summary(exam_id, tex_path=tex_path)
        except Exception as e:
            # don't fail upload; just record parse failure in meta preview_text
            # (or log it)
            pass

    # Parse for metadata
    # Preview / metadata
    if content_type == "questions_only":
        try:
            parts = parse_reference_tex(tex_path)
            keys = sorted([f"Q{k.qnum}{k.part}" for k in parts.values()])
            qnums = sorted({rp.qnum for rp in parts.values()})
            part_count = len(parts)
            q_count = len(qnums)

            preview_lines = []
            preview_lines.append("Canonical questions-only structure:")
            preview_lines.append("")
            for rp in list(parts.values())[:30]:
                clean_body = re.sub(r"\s+", " ", (rp.latex_body or "").strip())
                preview_lines.append(f"Q{rp.qnum}{rp.part}: {clean_body[:220]}")
            preview_text = "\n".join(preview_lines)
        except Exception:
            keys, preview_text = _preview_from_exam_structure(structure, tex_text_for_structure)
            q_count = structure.get("question_count") if structure else None
            part_count = structure.get("part_count") if structure else None
    else:
        try:
            parts = parse_reference_tex(tex_path)
            keys = sorted([f"Q{k.qnum}{k.part}" for k in parts.values()])
            qnums = sorted({rp.qnum for rp in parts.values()})
            part_count = len(parts)
            q_count = len(qnums)
            preview_lines = []
            for rp in list(parts.values())[:12]:
                clean_body = re.sub(r"\s+", " ", (rp.latex_body or "").strip())
                preview_lines.append(f"Q{rp.qnum}{rp.part}: {clean_body[:220]}")
            preview_text = "\n".join(preview_lines)
        except Exception as e:
            keys, q_count, part_count = [], None, None
            preview_text = f"Parse error: {e}"
    bundle_warnings = []
    bundle_structure_corrections = []

    try:
        bundle_path_for_meta = (
            uploads / "questions_only_bundle.json"
            if content_type == "questions_only"
            else uploads / "reference_bundle.json"
        )
        if bundle_path_for_meta.exists():
            bundle_meta = json.loads(bundle_path_for_meta.read_text(encoding="utf-8"))
            bundle_warnings = bundle_meta.get("warnings") or []
            bundle_structure_corrections = bundle_meta.get("structure_corrections") or []
    except Exception:
        bundle_warnings = []
        bundle_structure_corrections = []

    meta = {
        "exam_id": exam_id,
        "filename": filename,
        "original_filename": extra_meta.get("original_filename") or tex_file.filename,
        "content_type": content_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "q_count": q_count,
        "part_count": part_count,
        "keys_preview": keys[:50],
        "preview_text": preview_text[:4000],
        "source_kind": extra_meta.get("source_kind"),
        "ocr_used": extra_meta.get("ocr_used", False),
        "mathpix_mode": extra_meta.get("mathpix_mode"),
        "pdf_id": extra_meta.get("pdf_id"),
        "downloaded_ext": extra_meta.get("downloaded_ext"),
        "exam_structure_path": str(structure_path) if structure_path else None,
        "exam_structure_question_count": structure.get("question_count") if structure else None,
        "exam_structure_part_count": structure.get("part_count") if structure else None,
        "mathpix_text_path": extra_meta.get("mathpix_text_path"),
        "bundle_json_path": (
            str(uploads / "questions_only_bundle.json")
            if content_type == "questions_only"
            else str(uploads / "reference_bundle.json")
            if content_type == "reference"
            else None
        ),
        "bundle_warnings": bundle_warnings,
        "bundle_structure_corrections": bundle_structure_corrections,
    }

    meta_path = uploads / (filename + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(job_id, f"Saved metadata: {meta_path.name}")
    _progress(job_id, f"Done. Parsed questions: {q_count}, parts: {part_count}")

    if job_id:
        done(job_id)
    return meta

@router.get("/list")
def list_exam_files(
    exam_id: str,
    _session: dict = Depends(require_teacher),
):
    exam_id = require_safe_exam_id(exam_id)
    uploads = uploads_dir(exam_id)

    items = []
    for tex_path in sorted(uploads.glob("*.tex"), reverse=True):
        meta_path = uploads / (tex_path.name + ".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        items.append({
            "filename": tex_path.name,
            "content_type": meta.get("content_type"),
            "created_at": meta.get("created_at"),
            "q_count": meta.get("q_count"),
            "part_count": meta.get("part_count"),
            "source_kind": meta.get("source_kind"),
            "ocr_used": meta.get("ocr_used", False),
            "mathpix_mode": meta.get("mathpix_mode"),
            "exam_structure_question_count": meta.get("exam_structure_question_count"),
            "exam_structure_part_count": meta.get("exam_structure_part_count"),
        })

    return {"exam_id": exam_id, "items": items}

@router.get("/preview")
def preview_file(exam_id: str, filename: str, _session: dict = Depends(require_teacher)):
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    meta_path = uploads / (filename + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        return {
            "filename": filename,
            "content_type": meta.get("content_type"),
            "source_kind": meta.get("source_kind"),
            "keys": meta.get("keys_preview", []),
            "preview_text": meta.get("preview_text", ""),
            "q_count": meta.get("q_count"),
            "part_count": meta.get("part_count"),
            "exam_structure_question_count": meta.get("exam_structure_question_count"),
            "exam_structure_part_count": meta.get("exam_structure_part_count"),
            "ocr_used": meta.get("ocr_used", False),
        }

    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")

    try:
        parts = parse_reference_tex(tex_text)
        keys = sorted([f"Q{rp.qnum}{rp.part}" for rp in parts.values()])
        preview_lines = []
        for rp in list(parts.values())[:12]:
            title = re.sub(r"\s+", " ", (rp.title or "").strip())
            preview_lines.append(f"Q{rp.qnum}{rp.part}: {title[:120]}")
        preview_text = "\n".join(preview_lines)
    except Exception:
        keys = []
        preview_text = _strip_tex_wrapper_for_preview(tex_text)[:4000]

    return {
        "filename": filename,
        "keys": keys[:80],
        "preview_text": preview_text,
    }

@router.get("/raw")
def raw_file(exam_id: str, filename: str, _session: dict = Depends(require_teacher)):
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    text = tex_path.read_text(encoding="utf-8", errors="replace")

    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )

@router.delete("/delete")
def delete_file(exam_id: str, filename: str, _session: dict = Depends(require_teacher)):
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    meta_path = uploads / (filename + ".meta.json")

    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    tex_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    _cleanup_empty_exam_folder(exam_id)

    return {
        "ok": True,
        "deleted": filename,
        "exam_id": exam_id,
    }

@router.get("/exams")
def list_exams(_session: dict = Depends(require_teacher)):
    BANK_ROOT.mkdir(parents=True, exist_ok=True)

    exam_ids = []
    for p in sorted(BANK_ROOT.iterdir()):
        if not p.is_dir():
            continue

        # Only show exams that still contain at least one TeX solution/reference file.
        if any(p.rglob("*.tex")):
            exam_ids.append(p.name)

    return {"exam_ids": exam_ids}

@router.post("/exam/delete")
def delete_exam(req: ExamDeleteReq, _session: dict = Depends(require_teacher)):
    """Delete an entire exam_id folder (uploads + meta + files)."""
    exam_id = require_safe_exam_id(req.exam_id)
    d = exam_dir(exam_id)  # BANK_ROOT/exam_id

    if not d.exists():
        raise HTTPException(status_code=404, detail="exam_id not found")

    # Safety: refuse deleting the bank root by mistake
    if d.resolve() == BANK_ROOT.resolve():
        raise HTTPException(status_code=400, detail="Refusing to delete bank root")

    shutil.rmtree(d)
    return {"ok": True, "deleted": exam_id}




@router.post("/exam/rename")
def rename_exam(req: ExamRenameReq, _session: dict = Depends(require_teacher)):
    """Rename an exam_id folder (and update meta exam_id fields)."""
    old_id = require_safe_exam_id(req.old_exam_id)
    new_id = require_safe_exam_id(req.new_exam_id)

    if old_id == new_id:
        return {"ok": True, "renamed": False, "exam_id": old_id}

    old_dir = exam_dir(old_id)
    new_dir = exam_dir(new_id)

    if not old_dir.exists():
        raise HTTPException(status_code=404, detail="old exam_id not found")
    if new_dir.exists():
        raise HTTPException(status_code=409, detail="new exam_id already exists")

    old_dir.rename(new_dir)

    # Update stored metadata so preview/list show the new id
    for meta_path in new_dir.rglob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["exam_id"] = new_id
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return {"ok": True, "old_exam_id": old_id, "new_exam_id": new_id}