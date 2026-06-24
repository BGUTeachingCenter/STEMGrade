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
from common.tex.reference_tex import parse_reference_tex

from starlette.concurrency import run_in_threadpool

from core.ai_clients.ai_usage_logger import bind_usage_context, usage_log_dir
from core.ai_clients.ocr_client import OcrClientError, run_ocr
from core.ai_clients.gpt_client import GptClient
from core.ai_clients.google_client import GoogleClient
from schemas.ocr_response import OcrOptions, OcrResponse
from schemas.reference_bundle import (
    AnswersOnlyBundle,
    FullSolutionBundle,
    QuestionsOnlyBundle,
)
from services.solution_bank.answers_ocr import build_answers_ocr_result
from services.solution_bank.full_solution_service import (
    build_full_solution_from_questions_and_answers,
    full_solution_to_exam_structure,
)
from services.solution_bank.questions_answers_ocr import build_questions_answers_ocr_result
from services.solution_bank.questions_ocr import build_questions_ocr_result

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

AI_SOLUTION_FALLBACK_PROVIDERS = {
    "openai",
    "gpt",
    "chatgpt",
    "google",
    "gemini",
    "ai_studio",
    "aistudio",
}


BANK_CONTENT_TYPES = {
    "questions_only",
    "answers_only",
    "questions_answers",

    # Backward-compatible aliases:
    "reference",
    "full_solution",
}

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


def _canonical_content_type(content_type: str) -> str:
    """
    Normalize teacher/UI upload type names.

    Supported teacher-facing modes:
      - questions_only
      - answers_only
      - questions_answers

    Backward compatibility:
      - reference      -> answers_only
      - full_solution  -> questions_answers
    """
    ct = (content_type or "").strip().lower()

    aliases = {
        "questions": "questions_only",
        "question_only": "questions_only",
        "exam": "questions_only",

        "answers": "answers_only",
        "answer_only": "answers_only",
        "solutions_only": "answers_only",
        "solution_only": "answers_only",
        "reference": "answers_only",

        "questions_and_answers": "questions_answers",
        "question_answer": "questions_answers",
        "qa": "questions_answers",
        "full_solution": "questions_answers",
    }

    return aliases.get(ct, ct)


def _make_solution_bank_ai_client(provider: str, model: str = ""):
    """
    Return the AI client used for solution-bank JSON structuring/fallback.

    Mathpix is OCR-only here, so its downstream structuring remains GPT-backed.
    GPT/OpenAI and Google/Gemini uploads use the same provider for structuring
    and missing-answer fallback.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip() or None

    if provider in {"google", "gemini", "ai_studio", "aistudio"}:
        return GoogleClient(model=model)

    return GptClient(model=model)


def _answer_fallback_enabled_for_provider(provider: str) -> bool:
    return (provider or "").strip().lower() in AI_SOLUTION_FALLBACK_PROVIDERS


def _bundle_path(uploads: Path, name: str) -> Path:
    return uploads / name


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _count_full_solution(bundle: dict | FullSolutionBundle | None) -> tuple[int | None, int | None]:
    if bundle is None:
        return None, None

    try:
        full = bundle if isinstance(bundle, FullSolutionBundle) else FullSolutionBundle.model_validate(bundle)
        q_count = len(full.questions)
        part_count = sum(len(q.parts or []) for q in full.questions)
        return q_count, part_count
    except Exception:
        return None, None


def _gradeable_status_from_bundle(bundle: dict | FullSolutionBundle | None) -> str:
    """
    Coarse quality status for teacher/UI.

    clean:
      all submitted parts have an answer and no high-risk warnings/statuses.

    draft_review_required:
      full solution exists but some answers are AI-generated, uncertain,
      missing, weakly aligned, or otherwise require teacher review.

    not_gradeable:
      no full solution or no parts.
    """
    if bundle is None:
        return "not_gradeable"

    try:
        full = bundle if isinstance(bundle, FullSolutionBundle) else FullSolutionBundle.model_validate(bundle)
    except Exception:
        return "not_gradeable"

    total = 0
    answered = 0
    risky = False

    high_risk_phrases = [
        "AI fallback generated",
        "Answer generated by AI fallback",
        "No matching answer",
        "High missing-answer rate",
        "moderate confidence",
        "Teacher review",
        "draft",
        "uncertain",
    ]

    for q in full.questions:
        if not bool(getattr(q, "is_gradeable", True)) or getattr(q, "usage", "gradeable") != "gradeable":
            continue

        for p in q.parts:
            if not bool(getattr(p, "is_gradeable", getattr(q, "is_gradeable", True))):
                continue
            if getattr(p, "usage", getattr(q, "usage", "gradeable")) != "gradeable":
                continue

            total += 1
            if str(p.official_solution or "").strip() or str(p.expected_answer or "").strip():
                answered += 1

            if p.review_status and p.review_status != "ok":
                risky = True

            joined_warnings = "\n".join(str(w) for w in (p.warnings or []))
            if any(phrase.lower() in joined_warnings.lower() for phrase in high_risk_phrases):
                risky = True

    bundle_warnings = "\n".join(str(w) for w in (full.warnings or []))
    if any(phrase.lower() in bundle_warnings.lower() for phrase in high_risk_phrases):
        risky = True

    if total == 0 or answered == 0:
        return "not_gradeable"

    if answered < total or risky:
        return "draft_review_required"

    return "clean"


def _practice_counts_from_bundle(bundle: dict | FullSolutionBundle | None) -> tuple[int, int]:
    if bundle is None:
        return 0, 0

    try:
        full = bundle if isinstance(bundle, FullSolutionBundle) else FullSolutionBundle.model_validate(bundle)
    except Exception:
        return 0, 0

    q_count = 0
    part_count = 0

    for q in full.questions:
        q_is_practice = (
            not bool(getattr(q, "is_gradeable", True))
            or getattr(q, "usage", "gradeable") == "practice_feedback_only"
        )

        practice_parts = [
            p for p in q.parts
            if (
                q_is_practice
                or not bool(getattr(p, "is_gradeable", getattr(q, "is_gradeable", True)))
                or getattr(p, "usage", getattr(q, "usage", "gradeable")) == "practice_feedback_only"
            )
        ]

        if practice_parts:
            q_count += 1
            part_count += len(practice_parts)

    return q_count, part_count


def _build_and_save_full_solution_if_possible(
    *,
    uploads: Path,
    exam_id: str,
    source_name: str,
    job_id: str | None = None,
    enable_answer_fallback: bool = False,
    answer_fallback_client=None,
    answer_fallback_provider: str = "",
    answer_fallback_model: str = "",
    answer_fallback_raw_text: str = "",
) -> tuple[dict | None, str, dict | None]:
    """
    Try to build final full solution from existing questions_only + answers_only.

    Returns:
      (full_solution_bundle_dict_or_none, canonical_tex, exam_structure_or_none)
    """
    questions_path = uploads / "questions_only_bundle.json"
    answers_path = uploads / "answers_only_bundle.json"

    questions_dict = _read_json_if_exists(questions_path)
    answers_dict = _read_json_if_exists(answers_path)

    if not questions_dict or not answers_dict:
        return None, "", None

    _progress(job_id, "Both questions and answers are available. Building full solution...")

    full_result = build_full_solution_from_questions_and_answers(
        questions_bundle=QuestionsOnlyBundle.model_validate(questions_dict),
        answers_bundle=AnswersOnlyBundle.model_validate(answers_dict),
        exam_id=exam_id,
        source_name=source_name,
        out_dir=None,
        enable_answer_fallback=enable_answer_fallback,
        answer_fallback_client=answer_fallback_client,
        answer_fallback_provider=answer_fallback_provider,
        answer_fallback_model=answer_fallback_model,
        answer_fallback_raw_text=answer_fallback_raw_text,
    )

    full_dict = full_result.full_solution_bundle.model_dump()
    canonical_tex = full_result.canonical_tex
    structure = full_solution_to_exam_structure(full_result.full_solution_bundle)

    _write_json(uploads / "full_solution_bundle.json", full_dict)

    # Compatibility: existing grading/reference code expects reference_bundle.json.
    _write_json(uploads / "reference_bundle.json", full_dict)

    structure_path = uploads / "exam_structure.json"
    _write_json(structure_path, structure)

    tex_path = uploads / "reference_current.tex"
    tex_path.write_text(canonical_tex, encoding="utf-8")

    try:
        write_reference_summary(exam_id, tex_path=tex_path)
    except Exception:
        pass

    _progress(job_id, "Generated full_solution_bundle.json and reference_current.tex")

    return full_dict, canonical_tex, structure


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

            usage = str(q.get("usage") or "gradeable")
            label = "practice only" if usage == "practice_feedback_only" or q.get(
                "is_gradeable") is False else "gradeable"

            if parts:
                keys.extend([f"Q{qid}{p}" for p in parts])
                preview_lines.append(f"Question {qid}: parts {', '.join(parts)} [{label}]")
            else:
                keys.append(f"Q{qid}")
                preview_lines.append(f"Question {qid}: no parts detected [{label}]")

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
    """Upload a teacher file into the solution bank.

    content_type:
      - questions_only
      - answers_only
      - questions_answers
    """
    exam_id = require_safe_exam_id(exam_id)
    content_type = _canonical_content_type(content_type)

    if content_type not in {"questions_only", "answers_only", "questions_answers"}:
        raise HTTPException(
            status_code=400,
            detail="content_type must be questions_only, answers_only, or questions_answers",
        )

    # Attribute every OCR/AI usage record produced during this upload to this
    # exam + content_type, so data/ai_usage_logs can be aggregated per solution.
    bind_usage_context(
        exam_id=exam_id,
        content_type=content_type,
        bank_source=tex_file.filename,
    )

    if job_id:
        init_job(job_id)

    _progress(job_id, f"Starting upload for {content_type}: {tex_file.filename}")

    ocr_provider = (ocr_provider or "mathpix").strip().lower()
    if ocr_provider not in OCR_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail="ocr_provider must be mathpix, openai/gpt, or google/gemini",
        )

    solution_ai_client = _make_solution_bank_ai_client(ocr_provider, ocr_model)
    enable_answer_fallback = _answer_fallback_enabled_for_provider(ocr_provider)

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

    if content_type == "questions_only":
        filename = "questions_only_current.tex"
    elif content_type == "answers_only":
        filename = "answers_only_current.tex"
    else:
        filename = "questions_answers_current.tex"
    tex_path = uploads / filename
    tex_text_for_structure = raw.decode("utf-8", errors="replace")

    structure = None
    structure_path = None

    full_solution_bundle: dict | None = None
    full_solution_structure: dict | None = None
    final_reference_tex_generated = False

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

            _progress(job_id, "AI is organizing the exam into questions-only JSON...")

            questions_result = build_questions_ocr_result(
                ocr=ocr_response,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                out_dir=None,
                client=solution_ai_client,
            )

            questions_bundle = questions_result.questions_bundle.model_dump()

            bundle_path = uploads / "questions_only_bundle.json"
            _write_json(bundle_path, questions_bundle)

            _progress(job_id, "Saved questions_only_bundle.json")

            structure = questions_result.exam_structure
            structure_path = uploads / "exam_structure.json"
            _write_json(structure_path, structure)

            raw = questions_result.canonical_tex.encode("utf-8")

            _progress(job_id, "Generated canonical questions_only_current.tex")

            # If answers were uploaded earlier, now build the final solution.
            full_solution_bundle, full_tex, full_solution_structure = _build_and_save_full_solution_if_possible(
                uploads=uploads,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                job_id=job_id,
                enable_answer_fallback=enable_answer_fallback,
                answer_fallback_client=solution_ai_client,
                answer_fallback_provider=ocr_provider,
                answer_fallback_model=ocr_model or str(extra_meta.get("ocr_model") or ""),
                answer_fallback_raw_text=extra_meta.get("ocr_text") or "",
            )

            if full_solution_bundle and full_tex:
                final_reference_tex_generated = True

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

    elif content_type == "answers_only":
        try:
            ocr_response = extra_meta.get("ocr_response")
            if not isinstance(ocr_response, OcrResponse):
                ocr_response = OcrResponse(
                    provider="unknown",
                    input_kind="unknown",
                    source_filename=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    text=extra_meta.get("ocr_text") or extra_meta.get("mathpix_text") or tex_text_for_structure,
                )

            _progress(job_id, "AI is organizing the official answers into answers-only JSON...")

            existing_questions_bundle = _read_json_if_exists(uploads / "questions_only_bundle.json")

            answers_result = build_answers_ocr_result(
                ocr=ocr_response,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                out_dir=None,
                client=solution_ai_client,
                questions_bundle=existing_questions_bundle,
            )

            answers_bundle = answers_result.answers_bundle.model_dump()

            answers_bundle_path = uploads / "answers_only_bundle.json"
            _write_json(answers_bundle_path, answers_bundle)

            _progress(job_id, "Saved answers_only_bundle.json")

            # Save a light preview TeX for this upload itself.
            raw = tex_text_for_structure.encode("utf-8")

            full_solution_bundle, full_tex, full_solution_structure = _build_and_save_full_solution_if_possible(
                uploads=uploads,
                exam_id=exam_id,
                source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                job_id=job_id,
                enable_answer_fallback=enable_answer_fallback,
                answer_fallback_client=solution_ai_client,
                answer_fallback_provider=ocr_provider,
                answer_fallback_model=ocr_model or str(extra_meta.get("ocr_model") or ""),
                answer_fallback_raw_text=extra_meta.get("ocr_text") or "",
            )

            if full_solution_bundle and full_tex:
                final_reference_tex_generated = True
                structure = full_solution_structure
                structure_path = uploads / "exam_structure.json"

        except Exception as e:
            fail_path = uploads / "answers_only_ai_builder_error.txt"
            fail_path.write_text(str(e), encoding="utf-8")
            fail(job_id, f"Answers-only AI builder failed: {e}") if job_id else None

            raise HTTPException(
                status_code=500,
                detail=f"Answers-only AI builder failed. See {fail_path.name}.",
            )

    elif content_type == "questions_answers":
        try:
            ocr_response = extra_meta.get("ocr_response")
            if not isinstance(ocr_response, OcrResponse):
                ocr_response = OcrResponse(
                    provider="unknown",
                    input_kind="unknown",
                    source_filename=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    text=extra_meta.get("ocr_text") or extra_meta.get("mathpix_text") or tex_text_for_structure,
                )

            existing_questions_bundle = _read_json_if_exists(uploads / "questions_only_bundle.json")

            if existing_questions_bundle:
                _progress(
                    job_id,
                    "Locked questions_only_bundle.json exists. Treating this upload as an answer source and merging into the locked question structure...",
                )

                answers_result = build_answers_ocr_result(
                    ocr=ocr_response,
                    exam_id=exam_id,
                    source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    out_dir=None,
                    client=solution_ai_client,
                    questions_bundle=existing_questions_bundle,
                )

                answers_bundle = answers_result.answers_bundle.model_dump()
                _write_json(uploads / "answers_only_bundle.json", answers_bundle)

                full_solution_bundle, full_tex, full_solution_structure = _build_and_save_full_solution_if_possible(
                    uploads=uploads,
                    exam_id=exam_id,
                    source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",
                    job_id=job_id,
                    enable_answer_fallback=enable_answer_fallback,
                    answer_fallback_client=solution_ai_client,
                    answer_fallback_provider=ocr_provider,
                    answer_fallback_model=ocr_model or str(extra_meta.get("ocr_model") or ""),
                    answer_fallback_raw_text=extra_meta.get("ocr_text") or "",
                )

                if not full_solution_bundle or not full_tex:
                    raise RuntimeError("Locked-questions merge did not produce a full solution.")

                structure = full_solution_structure
                structure_path = uploads / "exam_structure.json"
                raw = full_tex.encode("utf-8")
                final_reference_tex_generated = True

                _progress(
                    job_id,
                    "Generated full solution from locked questions + uploaded answers. The answer file was not allowed to create new gradeable questions.",
                )


            else:

                _progress(

                    job_id,

                    "AI is organizing combined questions+answers into full solution JSON. "

                    "If that is too large, it will recover from questions-only extraction and generate draft answers...",

                )

                qa_result = build_questions_answers_ocr_result(

                    ocr=ocr_response,

                    exam_id=exam_id,

                    source_name=extra_meta.get("original_filename") or tex_file.filename or "upload",

                    out_dir=uploads,

                    client=solution_ai_client,

                    enable_answer_fallback=enable_answer_fallback,

                    answer_fallback_client=solution_ai_client,

                    answer_fallback_provider=ocr_provider,

                    answer_fallback_model=ocr_model or str(extra_meta.get("ocr_model") or ""),

                )

                qa_bundle = qa_result.questions_answers_bundle.model_dump()

                _write_json(uploads / "questions_answers_bundle.json", qa_bundle)

                if not qa_result.full_solution_bundle:
                    raise RuntimeError("questions_answers upload did not produce a full_solution_bundle")

                full_solution_bundle = qa_result.full_solution_bundle.model_dump()

                _write_json(uploads / "full_solution_bundle.json", full_solution_bundle)

                # Compatibility with existing grading/reference pipeline.

                _write_json(uploads / "reference_bundle.json", full_solution_bundle)

                structure = full_solution_to_exam_structure(qa_result.full_solution_bundle)

                structure_path = uploads / "exam_structure.json"

                _write_json(structure_path, structure)

                raw = qa_result.canonical_tex.encode("utf-8")

                final_reference_tex_generated = True

                _progress(job_id, "Generated full_solution_bundle.json and canonical reference_current.tex")
        except Exception as e:
            fail_path = uploads / "questions_answers_ai_builder_error.txt"
            fail_path.write_text(str(e), encoding="utf-8")
            fail(job_id, f"Questions+answers AI builder failed: {e}") if job_id else None

            raise HTTPException(
                status_code=500,
                detail=f"Questions+answers AI builder failed. See {fail_path.name}.",
            )

    tex_path.write_bytes(raw)
    tex_text_for_structure = raw.decode("utf-8", errors="replace")

    # If this upload produced a final full solution, also write/refresh reference_current.tex.
    if content_type == "questions_answers" and final_reference_tex_generated:
        reference_tex_path = uploads / "reference_current.tex"
        reference_tex_path.write_bytes(raw)
        tex_path = reference_tex_path
        filename = "reference_current.tex"
        tex_text_for_structure = raw.decode("utf-8", errors="replace")

    if content_type in {"questions_only", "answers_only"} and final_reference_tex_generated:
        # The current upload file is still saved as questions_only_current.tex / answers_only_current.tex,
        # but the gradeable reference file is reference_current.tex.
        pass

    # create/update reference_summary.json for fast matching once a full solution exists
    reference_tex_path = uploads / "reference_current.tex"
    if reference_tex_path.exists():
        try:
            write_reference_summary(exam_id, tex_path=reference_tex_path)
        except Exception:
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
        if (uploads / "full_solution_bundle.json").exists():
            bundle_path_for_meta = uploads / "full_solution_bundle.json"
        elif content_type == "questions_only":
            bundle_path_for_meta = uploads / "questions_only_bundle.json"
        elif content_type == "answers_only":
            bundle_path_for_meta = uploads / "answers_only_bundle.json"
        elif content_type == "questions_answers":
            bundle_path_for_meta = uploads / "questions_answers_bundle.json"
        else:
            bundle_path_for_meta = uploads / "reference_bundle.json"

        if bundle_path_for_meta.exists():
            bundle_meta = json.loads(bundle_path_for_meta.read_text(encoding="utf-8"))
            bundle_warnings = bundle_meta.get("warnings") or []
            bundle_structure_corrections = bundle_meta.get("structure_corrections") or []
    except Exception:
        bundle_warnings = []
        bundle_structure_corrections = []

    gradeable_status = "not_gradeable"
    try:
        if (uploads / "full_solution_bundle.json").exists():
            gradeable_status = _gradeable_status_from_bundle(
                json.loads((uploads / "full_solution_bundle.json").read_text(encoding="utf-8"))
            )
    except Exception:
        gradeable_status = "not_gradeable"

    # Live token feedback for this upload: OCR tokens (when an AI OCR provider
    # was used) + the AI structuring/fallback tokens accumulated on the
    # solution_ai_client. The authoritative per-call record is written to
    # data/ai_usage_logs and aggregated per solution by /token_usage; this is
    # only used for the progress message below.
    ocr_tokens = 0
    try:
        ocr_response_for_usage = extra_meta.get("ocr_response")
        if isinstance(ocr_response_for_usage, OcrResponse):
            ocr_tokens = int(ocr_response_for_usage.usage.total_tokens or 0)
    except Exception:
        ocr_tokens = 0

    ai_tokens = 0
    try:
        ai_tokens = int(getattr(solution_ai_client, "total_tokens", 0) or 0)
    except Exception:
        ai_tokens = 0

    total_tokens = ocr_tokens + ai_tokens

    _progress(
        job_id,
        f"Token usage — OCR: {ocr_tokens}, AI structuring: {ai_tokens}, total: {total_tokens}",
    )

    practice_question_count = 0
    practice_part_count = 0
    try:
        if (uploads / "full_solution_bundle.json").exists():
            practice_question_count, practice_part_count = _practice_counts_from_bundle(
                json.loads((uploads / "full_solution_bundle.json").read_text(encoding="utf-8"))
            )
        elif (uploads / "questions_only_bundle.json").exists():
            q_only = QuestionsOnlyBundle.model_validate(
                json.loads((uploads / "questions_only_bundle.json").read_text(encoding="utf-8"))
            )
            practice_question_count = sum(
                1
                for q in q_only.questions
                if not bool(getattr(q, "is_gradeable", True))
                or getattr(q, "usage", "gradeable") == "practice_feedback_only"
            )
            practice_part_count = sum(
                1
                for q in q_only.questions
                for p in q.parts
                if not bool(getattr(p, "is_gradeable", getattr(q, "is_gradeable", True)))
                or getattr(p, "usage", getattr(q, "usage", "gradeable")) == "practice_feedback_only"
            )
    except Exception:
        practice_question_count = 0
        practice_part_count = 0

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
            str(uploads / "full_solution_bundle.json")
            if (uploads / "full_solution_bundle.json").exists()
            else str(uploads / "questions_only_bundle.json")
            if content_type == "questions_only"
            else str(uploads / "answers_only_bundle.json")
            if content_type == "answers_only"
            else str(uploads / "questions_answers_bundle.json")
            if content_type == "questions_answers"
            else None
        ),
        "full_solution_available": (uploads / "full_solution_bundle.json").exists(),
        "reference_tex_available": (uploads / "reference_current.tex").exists(),
        "gradeable_status": gradeable_status,
        "practice_question_count": practice_question_count,
        "practice_part_count": practice_part_count,
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
            "full_solution_available": meta.get("full_solution_available"),
            "reference_tex_available": meta.get("reference_tex_available"),
            "gradeable_status": meta.get("gradeable_status"),
            "practice_question_count": meta.get("practice_question_count"),
            "practice_part_count": meta.get("practice_part_count"),
        })

    return {"exam_id": exam_id, "items": items}

@router.get("/token_usage")
def bank_token_usage(_session: dict = Depends(require_teacher)):
    """
    Aggregate token usage per solution from data/ai_usage_logs/*.jsonl.

    Each line is one OCR/AI call. Records produced during a solution-bank
    upload carry exam_id + content_type (see bind_usage_context). Records
    without an exam_id (e.g. student grading, or uploads logged before this
    attribution existed) are ignored here.

    Returns per-exam totals plus an OCR vs AI (structuring) split, and a
    per-content_type breakdown for finer display.
    """
    log_dir = usage_log_dir()

    by_exam: dict[str, dict] = {}

    for log_path in sorted(log_dir.glob("ai_usage_*.jsonl")):
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            exam_id = (rec.get("exam_id") or "").strip()
            if not exam_id:
                continue

            try:
                tok = int(rec.get("total_tokens") or 0)
            except Exception:
                tok = 0

            task = str(rec.get("task") or "")
            content_type = (rec.get("content_type") or "").strip() or "unknown"

            entry = by_exam.setdefault(
                exam_id,
                {
                    "exam_id": exam_id,
                    "total_tokens": 0,
                    "ocr_tokens": 0,
                    "ai_tokens": 0,
                    "calls": 0,
                    "by_content_type": {},
                },
            )

            entry["total_tokens"] += tok
            entry["calls"] += 1
            if task == "ocr":
                entry["ocr_tokens"] += tok
            else:
                entry["ai_tokens"] += tok

            ct = entry["by_content_type"].setdefault(
                content_type,
                {"total_tokens": 0, "ocr_tokens": 0, "ai_tokens": 0, "calls": 0},
            )
            ct["total_tokens"] += tok
            ct["calls"] += 1
            if task == "ocr":
                ct["ocr_tokens"] += tok
            else:
                ct["ai_tokens"] += tok

    return {"by_exam": by_exam}

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
            "full_solution_available": meta.get("full_solution_available"),
            "reference_tex_available": meta.get("reference_tex_available"),
            "gradeable_status": meta.get("gradeable_status"),
            "practice_question_count": meta.get("practice_question_count"),
            "practice_part_count": meta.get("practice_part_count"),
            "bundle_json_path": meta.get("bundle_json_path"),
            "bundle_warnings": meta.get("bundle_warnings", []),
            "bundle_structure_corrections": meta.get("bundle_structure_corrections", []),
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