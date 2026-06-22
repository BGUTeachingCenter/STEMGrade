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
from fastapi.responses import PlainTextResponse

from core.security import require_teacher
from core.storage import require_safe_exam_id, require_safe_filename, uploads_dir, write_reference_summary
from core.config import BANK_ROOT
from services.file_handling.reference_tex import parse_reference_tex

from starlette.concurrency import run_in_threadpool

from core.ai_clients.ocr_client import OcrClientError, run_ocr
from schemas.ocr_response import OcrOptions, OcrResponse
from schemas.reference_bundle import ReferenceBundle
from services.ocr_services.questions_ocr import build_questions_ocr_result
from services.ocr_services.reference_solution_ocr import build_reference_solution_ocr_result

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
    Remove the whole exam folder after the last TeX file is deleted.

    The upload folder can still contain generated files such as:
      - reference_bundle.json
      - questions_only_bundle.json
      - Mathpix raw/text files
      - originals/
      - reference_summary.json

    If there are no .tex files left, the exam is no longer usable in the
    solution bank, so remove the whole exam folder.
    """
    d = exam_dir(exam_id)

    if not d.exists():
        return

    # If any TeX file still exists, keep the exam folder.
    if any(d.rglob("*.tex")):
        return

    # No usable reference/questions files remain, so remove the whole folder.
    try:
        shutil.rmtree(d)
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

OCR_PROVIDERS = {"mathpix", "openai", "gpt", "chatgpt", "google", "gemini", "ai_studio", "aistudio"}


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
    ocr_provider: str = "mathpix",
    ocr_model: str = "",
) -> tuple[bytes, dict]:
    """
    Convert either a TeX upload or an OCR-able upload into temporary TeX bytes.

    Important:
      - This function does universal extraction only.
      - It does NOT decide whether the file is questions/reference/student work.
      - Task-specific interpretation happens later via services/ocr_services.
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

    uploads = uploads_dir(exam_id)

    # ------------------------------------------------------------
    # Case 1: Teacher uploaded TeX/text directly.
    # Treat it as a provider-neutral OCR response with provider="tex".
    # ------------------------------------------------------------
    if suffix in TEX_SUFFIXES:
        text = raw.decode("utf-8", errors="replace")

        ocr_response = OcrResponse(
            provider="tex",
            model=None,
            input_kind="tex" if suffix == ".tex" else "text",
            source_filename=safe_original_name,
            source_path="",
            text=text,
            pages=[],
            usage=None,  # allowed only if your schema has Optional; otherwise remove this line
        )

        return raw, {
            "source_kind": "tex",
            "original_filename": safe_original_name,
            "ocr_used": False,
            "pdf_text_layer_used": False,
            "ocr_provider": "tex",
            "ocr_model": None,
            "ocr_response": ocr_response,
            "ocr_response_path": None,
            "ocr_raw_path": None,
            "ocr_text_path": None,
            "ocr_text": text,
            "mathpix_text": text,
            "mathpix_text_path": None,
            "mathpix_mode": None,
            "pdf_id": None,
            "downloaded_ext": None,
            "openai_text_path": None,
            "openai_model": None,
            "openai_response_id": None,
        }

    # ------------------------------------------------------------
    # Case 2: PDF/image upload.
    # Use universal OCR client.
    # ------------------------------------------------------------
    ocr_provider = (ocr_provider or "mathpix").strip().lower()
    if ocr_provider not in OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported OCR provider: {ocr_provider}.",
        )

    originals_dir = uploads / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    original_path = originals_dir / safe_original_name
    original_path.write_bytes(raw)

    try:
        ocr_response = await run_in_threadpool(
            run_ocr,
            file_path=original_path,
            provider=ocr_provider,
            model=ocr_model or None,
            options=OcrOptions(
                temperature=0.0,
                max_output_tokens=12000,
                timeout_s=300,
                language_hint="hebrew",
                preserve_math=True,
                preserve_layout=True,
                include_line_data=True,
            ),
        )
    except OcrClientError as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    ocr_text = ocr_response.primary_text()

    provider = str(ocr_response.provider or ocr_provider)
    provider_label = provider.replace("/", "_")

    ocr_response_path = uploads / f"{content_type}_{provider_label}_ocr_response.json"
    ocr_response_path.write_text(
        json.dumps(ocr_response.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ocr_text_path = uploads / f"{content_type}_{provider_label}_text.txt"
    ocr_text_path.write_text(ocr_text, encoding="utf-8")

    # Remove markdown image links. They are useful for debugging,
    # but not valid inside temporary TeX.
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
            rf"% OCR generated for solution bank via {provider}. Source: {safe_original_name}",
            "",
            ocr_text_for_tex.strip(),
            "",
            r"\end{document}",
            "",
        ]
    )

    return tex.encode("utf-8"), {
        "source_kind": provider,
        "original_filename": safe_original_name,
        "ocr_used": True,
        "ocr_provider": provider,
        "ocr_model": ocr_response.model,
        "ocr_response": ocr_response,
        "ocr_response_path": str(ocr_response_path),
        "ocr_raw_path": str(ocr_response_path),
        "ocr_text_path": str(ocr_text_path),
        "ocr_text": ocr_text,

        # Compatibility with old builder naming.
        "mathpix_text": ocr_text,

        # Compatibility / UI fields.
        "mathpix_text_path": str(ocr_text_path) if provider == "mathpix" else None,
        "mathpix_mode": ocr_response.provider_mode if provider == "mathpix" else None,
        "pdf_id": ocr_response.provider_document_id if provider == "mathpix" else None,
        "downloaded_ext": (ocr_response.raw or {}).get("downloaded_ext") or "txt",

        "openai_text_path": str(ocr_text_path) if provider == "openai" else None,
        "openai_model": ocr_response.model if provider == "openai" else None,
        "openai_response_id": ocr_response.response_id if provider == "openai" else None,
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
    ocr_provider: str = Form("mathpix"),
    ocr_model: str = Form(""),
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

    ocr_provider = (ocr_provider or "mathpix").strip().lower()
    if ocr_provider not in OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail="ocr_provider must be mathpix or openai",
        )

    _progress(job_id, f"Extracting file with {ocr_provider} if OCR is needed...")

    raw, extra_meta = await _bank_upload_to_tex(
        upload=tex_file,
        exam_id=exam_id,
        content_type=content_type,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
    )

    _progress(
        job_id,
        (
            f"Extraction complete. Source: {extra_meta.get('source_kind')}, "
            f"provider: {extra_meta.get('ocr_provider') or 'none'}, "
            f"model: {extra_meta.get('ocr_model') or extra_meta.get('openai_model') or 'tex'}, "
            f"mode: {extra_meta.get('mathpix_mode') or 'standard'}"
        ),
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
            ocr_response = extra_meta.get("ocr_response")
            if not isinstance(ocr_response, OcrResponse):
                ocr_response = OcrResponse(
                    provider="unknown",
                    input_kind="unknown",
                    source_filename=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    text=extra_meta.get("ocr_text") or extra_meta.get("mathpix_text") or tex_text_for_structure,
                )

            _progress(job_id, "AI is organizing the exam into question/part JSON...")

            questions_result = build_questions_ocr_result(
                ocr=ocr_response,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                out_dir=None,
            )

            questions_bundle = questions_result.questions_bundle.model_dump()

            bundle_path = uploads / "questions_only_bundle.json"
            bundle_path.write_text(
                json.dumps(questions_bundle, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            _progress(job_id, "Saved questions_only_bundle.json")

            structure = questions_result.exam_structure
            structure_path = uploads / "exam_structure.json"
            structure_path.write_text(
                json.dumps(structure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            raw = questions_result.canonical_tex.encode("utf-8")

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
            questions_bundle_dict = json.loads(
                questions_bundle_path.read_text(encoding="utf-8")
            )

            questions_bundle = ReferenceBundle.model_validate(questions_bundle_dict)

            ocr_response = extra_meta.get("ocr_response")
            if not isinstance(ocr_response, OcrResponse):
                ocr_response = OcrResponse(
                    provider="unknown",
                    input_kind="unknown",
                    source_filename=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    text=extra_meta.get("ocr_text") or extra_meta.get("mathpix_text") or tex_text_for_structure,
                )

            _progress(job_id, "AI is aligning official solutions to the exam structure...")

            reference_result = build_reference_solution_ocr_result(
                ocr=ocr_response,
                questions_bundle=questions_bundle,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                out_dir=None,
            )

            reference_bundle = reference_result.reference_bundle.model_dump()

            reference_bundle_path = uploads / "reference_bundle.json"
            reference_bundle_path.write_text(
                json.dumps(reference_bundle, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            _progress(job_id, "Saved reference_bundle.json")

            if reference_result.high_confidence_corrections_count:
                corrected_questions_path = uploads / "questions_only_bundle_corrected_by_reference.json"
                corrected_questions_path.write_text(
                    json.dumps(reference_bundle, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                corrected_structure = {
                    "questions": [
                        {
                            "question_id": str(q.get("question_id", "")),
                            "parts": [
                                p.get("part", "")
                                for p in (q.get("parts") or [])
                                if p.get("part")
                            ],
                        }
                        for q in reference_bundle.get("questions", [])
                    ]
                }
                corrected_structure["question_count"] = len(corrected_structure["questions"])
                corrected_structure["part_count"] = sum(
                    len(q.get("parts") or [])
                    for q in corrected_structure["questions"]
                )

                corrected_structure_path = uploads / "exam_structure_corrected_by_reference.json"
                corrected_structure_path.write_text(
                    json.dumps(corrected_structure, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                _progress(
                    job_id,
                    (
                        "Saved corrected structure from reference upload "
                        f"({reference_result.high_confidence_corrections_count} high-confidence corrections)."
                    ),
                )

            raw = reference_result.canonical_tex.encode("utf-8")

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
        "ocr_provider": extra_meta.get("ocr_provider"),
        "ocr_model": extra_meta.get("ocr_model"),
        "ocr_response_path": extra_meta.get("ocr_response_path"),
        "ocr_text_path": extra_meta.get("ocr_text_path"),
        "ocr_raw_path": extra_meta.get("ocr_raw_path"),

        # Legacy/UI compatibility fields
        "mathpix_mode": extra_meta.get("mathpix_mode"),
        "pdf_id": extra_meta.get("pdf_id"),
        "downloaded_ext": extra_meta.get("downloaded_ext"),
        "openai_model": extra_meta.get("openai_model"),
        "openai_response_id": extra_meta.get("openai_response_id"),
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
            "ocr_provider": meta.get("ocr_provider"),
            "ocr_model": meta.get("ocr_model"),
            "mathpix_mode": meta.get("mathpix_mode"),
            "openai_model": meta.get("openai_model"),
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
            "ocr_provider": meta.get("ocr_provider"),
            "ocr_model": meta.get("ocr_model"),
            "ocr_response_path": meta.get("ocr_response_path"),
            "mathpix_mode": meta.get("mathpix_mode"),
            "openai_model": meta.get("openai_model"),
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