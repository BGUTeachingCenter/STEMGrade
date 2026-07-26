from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from core.security import require_session
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


def _student_code(session: dict) -> str:
    if session.get("role") != "student":
        raise HTTPException(
            status_code=403,
            detail="Only students can access saved work files.",
        )

    code = str(session.get("sub") or "").strip()
    if not code:
        raise HTTPException(status_code=403, detail="Missing student code in session.")

    return code


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


def _page_url(work_id: str, filename: str, page_number: int) -> str:
    return "/routes/student/work_page?" + urlencode(
        {
            "work_id": work_id,
            "filename": filename,
            "page_number": page_number,
        }
    )


@router.get("/work_file")
def student_work_file(
    work_id: str,
    filename: str,
    _session: dict = Depends(require_session),
):
    path = resolve_student_work_file(_student_code(_session), work_id, filename)
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
    _session: dict = Depends(require_session),
):
    path = resolve_student_work_file(_student_code(_session), work_id, filename)
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
                    "image_url": _page_url(work_id, filename, 1),
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
                        "image_url": _page_url(work_id, filename, page_number),
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
    _session: dict = Depends(require_session),
):
    path = resolve_student_work_file(_student_code(_session), work_id, filename)
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
