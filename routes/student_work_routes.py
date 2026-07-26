from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from core.security import require_session
from services.student_access import get_student_code_record
from services.student_work_store import resolve_student_work_file

router = APIRouter(prefix="/routes/student", tags=["student-work"])

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _session_student_code(
    session: dict,
    requested_student_code: str = "",
) -> str:
    role = str(session.get("role") or "").strip()

    if role == "student":
        code = str(session.get("sub") or "").strip()

        if not code:
            raise HTTPException(
                status_code=403,
                detail="Missing student code in session.",
            )

        if (
            requested_student_code
            and requested_student_code != code
        ):
            raise HTTPException(
                status_code=403,
                detail="Students can only access their own saved work.",
            )

        return code

    if role == "teacher":
        teacher_id = str(
            session.get("teacher_id")
            or session.get("sub")
            or ""
        ).strip()

        code = str(requested_student_code or "").strip()

        if not teacher_id or teacher_id == "teacher":
            raise HTTPException(
                status_code=403,
                detail="Missing teacher profile in session.",
            )

        if not code:
            raise HTTPException(
                status_code=400,
                detail="student_code is required for teacher review.",
            )

        record = get_student_code_record(code) or {}

        if str(record.get("teacher_id") or "").strip() != teacher_id:
            raise HTTPException(
                status_code=404,
                detail="Student work not found.",
            )

        return code

    raise HTTPException(
        status_code=403,
        detail="This session cannot access saved student work.",
    )


def _authorized_work_file(
    session: dict,
    requested_student_code: str,
    work_id: str,
    filename: str,
) -> tuple[str, Path]:
    student_code = _session_student_code(
        session,
        requested_student_code,
    )

    path = resolve_student_work_file(
        student_code,
        work_id,
        filename,
    )

    if session.get("role") == "teacher":
        teacher_id = str(
            session.get("teacher_id")
            or session.get("sub")
            or ""
        ).strip()

        metadata_path = resolve_student_work_file(
            student_code,
            work_id,
            "metadata.json",
        )

        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Saved student-work metadata could not be read.",
            ) from exc

        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=500,
                detail="Saved student-work metadata is invalid.",
            )

        metadata_teacher_id = str(
            metadata.get("teacher_id")
            or ""
        ).strip()

        metadata_student_code = str(
            metadata.get("student_code")
            or ""
        ).strip()

        if (
            metadata_teacher_id != teacher_id
            or metadata_student_code != student_code
        ):
            raise HTTPException(
                status_code=404,
                detail="Student work not found.",
            )

    return student_code, path


def _pymupdf():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="PyMuPDF is required for PDF page rendering.",
            ) from exc

    return fitz


def _page_url(
    work_id: str,
    filename: str,
    page_number: int,
    student_code: str = "",
) -> str:
    query = {
        "work_id": work_id,
        "filename": filename,
        "page_number": page_number,
    }

    if student_code:
        query["student_code"] = student_code

    return "/routes/student/work_page?" + urlencode(query)


@router.get("/work_file")
def student_work_file(
    work_id: str,
    filename: str,
    student_code: str = "",
    _session: dict = Depends(require_session),
):
    _resolved_code, path = _authorized_work_file(
        _session,
        student_code,
        work_id,
        filename,
    )
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".json":
        media_type = "application/json"
    elif suffix == ".tex":
        media_type = "application/x-tex; charset=utf-8"
    else:
        media_type = _IMAGE_MEDIA_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(path=str(path), media_type=media_type, filename=path.name)


@router.get("/work_manifest")
def student_work_manifest(
    work_id: str,
    filename: str,
    student_code: str = "",
    _session: dict = Depends(require_session),
):
    resolved_code, path = _authorized_work_file(
        _session,
        student_code,
        work_id,
        filename,
    )

    page_student_code = (
        resolved_code
        if _session.get("role") == "teacher"
        else ""
    )

    suffix = path.suffix.lower()

    if suffix in _IMAGE_MEDIA_TYPES:
        return {
            "ok": True,
            "kind": "image",
            "filename": path.name,
            "pages": [
                {
                    "page_number": 1,
                    "width": None,
                    "height": None,
                    "image_url": _page_url(
                        work_id,
                        filename,
                        1,
                        page_student_code,
                    ),
                }
            ],
        }

    if suffix != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF and image submissions support the visual overlay.",
        )

    fitz = _pymupdf()

    try:
        with fitz.open(str(path)) as document:
            if getattr(document, "needs_pass", False):
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs cannot be rendered.",
                )

            pages = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                rect = page.rect
                page_number = page_index + 1
                pages.append(
                    {
                        "page_number": page_number,
                        "width": float(rect.width),
                        "height": float(rect.height),
                        "image_url": _page_url(
                            work_id,
                            filename,
                            page_number,
                            page_student_code,
                        ),
                    }
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the submitted PDF: {exc}",
        ) from exc

    return {
        "ok": True,
        "kind": "pdf",
        "filename": path.name,
        "pages": pages,
    }


@router.get("/work_page")
def student_work_page(
    work_id: str,
    filename: str,
    page_number: int = 1,
    scale: float = 1.7,
    student_code: str = "",
    _session: dict = Depends(require_session),
):
    _resolved_code, path = _authorized_work_file(
        _session,
        student_code,
        work_id,
        filename,
    )
    suffix = path.suffix.lower()

    if suffix in _IMAGE_MEDIA_TYPES:
        if page_number != 1:
            raise HTTPException(status_code=404, detail="Image page not found.")

        return Response(
            content=path.read_bytes(),
            media_type=_IMAGE_MEDIA_TYPES[suffix],
            headers={"Cache-Control": "private, max-age=300"},
        )

    if suffix != ".pdf":
        raise HTTPException(status_code=415, detail="This file cannot be rendered as pages.")

    fitz = _pymupdf()

    try:
        with fitz.open(str(path)) as document:
            if page_number < 1 or page_number > document.page_count:
                raise HTTPException(status_code=404, detail="PDF page not found.")

            page = document.load_page(page_number - 1)
            rect = page.rect
            safe_scale = max(1.0, min(float(scale), 2.25))
            safe_scale = min(
                safe_scale,
                3200.0 / max(float(rect.width), 1.0),
                4200.0 / max(float(rect.height), 1.0),
            )
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(safe_scale, safe_scale),
                alpha=False,
            )
            png_bytes = pixmap.tobytes("png")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not render PDF page {page_number}: {exc}",
        ) from exc

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )
