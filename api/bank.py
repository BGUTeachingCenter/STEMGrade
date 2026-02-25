# api/bank.py
from __future__ import annotations

import json
import re
from datetime import datetime
import shutil
from pydantic import BaseModel
from core.storage import exam_dir  # add this import

from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from core.security import require_teacher_password
from core.storage import require_safe_exam_id, require_safe_filename, uploads_dir, write_reference_summary
from core.config import BANK_ROOT
from grader.reference_tex import parse_reference_tex

router = APIRouter(prefix="/api/bank", tags=["bank"])

@router.post("/upload")
async def upload_to_bank(
    exam_id: str = Form(...),
    content_type: str = Form(...),
    tex_file: UploadFile = File(...),
    x_teacher_password: str | None = Header(None),
):
    """Upload a teacher TeX file into the solution bank.

    - content_type: "reference" or "questions_only"
    - saves file as reference_current.tex / questions_only_current.tex
    - writes a lightweight .meta.json for fast preview/list
    """
    require_teacher_password(x_teacher_password)

    exam_id = require_safe_exam_id(exam_id)
    if content_type not in {"reference", "questions_only"}:
        raise HTTPException(status_code=400, detail="content_type must be reference or questions_only")

    raw = await tex_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    uploads = uploads_dir(exam_id)

    suffix = "reference" if content_type == "reference" else "questions_only"
    filename = f"{suffix}_current.tex"
    tex_path = uploads / filename
    tex_path.write_bytes(raw)
    # create/update reference_summary.json for fast matching
    if content_type == "reference":
        try:
            write_reference_summary(exam_id, tex_path=tex_path)
        except Exception as e:
            # don't fail upload; just record parse failure in meta preview_text
            # (or log it)
            pass

    # Parse for metadata
    try:
        parts = parse_reference_tex(tex_path)
        keys = sorted([f"Q{k.qnum}{k.part}" for k in parts.values()])
        qnums = sorted({rp.qnum for rp in parts.values()})
        part_count = len(parts)
        q_count = len(qnums)
        preview_lines = []
        for rp in list(parts.values())[:12]:
            title = re.sub(r"\s+", " ", (rp.title or "").strip())
            preview_lines.append(f"Q{rp.qnum}{rp.part}: {title[:120]}")
        preview_text = "\n".join(preview_lines)
    except Exception as e:
        keys, q_count, part_count = [], None, None
        preview_text = f"Parse error: {e}"

    meta = {
        "exam_id": exam_id,
        "filename": filename,
        "original_filename": tex_file.filename,
        "content_type": content_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "q_count": q_count,
        "part_count": part_count,
        "keys_preview": keys[:50],
        "preview_text": preview_text[:4000],
    }

    meta_path = uploads / (filename + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

@router.get("/list")
def list_exam_files(
    exam_id: str,
    x_teacher_password: str | None = Header(None),
):
    require_teacher_password(x_teacher_password)
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
        })

    return {"exam_id": exam_id, "items": items}

@router.get("/preview")
def preview_file(exam_id: str, filename: str, x_teacher_password: str | None = Header(None)):
    require_teacher_password(x_teacher_password)
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    meta_path = uploads / (filename + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"filename": filename, "keys": meta.get("keys_preview", []), "preview_text": meta.get("preview_text", "")}

    parts = parse_reference_tex(tex_path)
    keys = sorted([f"Q{rp.qnum}{rp.part}" for rp in parts.values()])
    preview_lines = []
    for rp in list(parts.values())[:12]:
        title = re.sub(r"\s+", " ", (rp.title or "").strip())
        preview_lines.append(f"Q{rp.qnum}{rp.part}: {title[:120]}")
    return {"filename": filename, "keys": keys[:50], "preview_text": "\n".join(preview_lines)}

@router.get("/raw")
def raw_file(exam_id: str, filename: str, x_teacher_password: str | None = Header(None)):
    require_teacher_password(x_teacher_password)
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(str(tex_path), media_type="text/plain; charset=utf-8", filename=filename)

@router.delete("/delete")
def delete_file(exam_id: str, filename: str, x_teacher_password: str | None = Header(None)):
    require_teacher_password(x_teacher_password)
    exam_id = require_safe_exam_id(exam_id)
    filename = require_safe_filename(filename)

    uploads = uploads_dir(exam_id)
    tex_path = uploads / filename
    meta_path = uploads / (filename + ".meta.json")

    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    tex_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return PlainTextResponse("Deleted")

@router.get("/exams")
def list_exams(x_teacher_password: str | None = Header(None)):
    require_teacher_password(x_teacher_password)
    BANK_ROOT.mkdir(parents=True, exist_ok=True)

    exam_ids = sorted([p.name for p in BANK_ROOT.iterdir() if p.is_dir()])
    return {"exam_ids": exam_ids}

class ExamDeleteReq(BaseModel):
    exam_id: str

@router.post("/exam/delete")
def delete_exam(req: ExamDeleteReq, x_teacher_password: str | None = Header(None)):
    """Delete an entire exam_id folder (uploads + meta + files)."""
    require_teacher_password(x_teacher_password)

    exam_id = require_safe_exam_id(req.exam_id)
    d = exam_dir(exam_id)  # BANK_ROOT/exam_id

    if not d.exists():
        raise HTTPException(status_code=404, detail="exam_id not found")

    # Safety: refuse deleting the bank root by mistake
    if d.resolve() == BANK_ROOT.resolve():
        raise HTTPException(status_code=400, detail="Refusing to delete bank root")

    shutil.rmtree(d)
    return {"ok": True, "deleted": exam_id}


class ExamRenameReq(BaseModel):
    old_exam_id: str
    new_exam_id: str

@router.post("/exam/rename")
def rename_exam(req: ExamRenameReq, x_teacher_password: str | None = Header(None)):
    """Rename an exam_id folder (and update meta exam_id fields)."""
    require_teacher_password(x_teacher_password)

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