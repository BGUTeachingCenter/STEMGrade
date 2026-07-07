from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from core.security import require_session
from services.student_work_store import resolve_student_work_file

router = APIRouter(prefix="/routes/student", tags=["student-work"])


@router.get("/work_file")
def student_work_file(
    work_id: str,
    filename: str,
    _session: dict = Depends(require_session),
):
    if _session.get("role") != "student":
        raise HTTPException(status_code=403, detail="Only students can download their saved work files.")

    student_code = str(_session.get("sub") or "").strip()
    if not student_code:
        raise HTTPException(status_code=403, detail="Missing student code in session.")

    path = resolve_student_work_file(student_code, work_id, filename)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".json":
        media_type = "application/json"
    elif suffix == ".tex":
        media_type = "application/x-tex; charset=utf-8"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=path.name,
    )