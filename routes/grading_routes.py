# routes/grading_routes.py
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from time import perf_counter
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from routes.progress import done, fail, init_job, push
from routes.student_log_routes import log_student_submission
from core.config import BANK_ROOT, FIXED_FONT, RUNS_ROOT
from core.storage import bind_bank_root, require_safe_exam_id, teacher_bank_root, write_reference_summary
from core.debug import create_debug_trace, write_debug_log
from core.security import require_session
from services.student_grading.grading.grader_payloads import grade_payload_manifest
from services.student_grading.grading.payloads import build_payloads
from services.student_grading.grading.solution_bank_matcher import pick_reference_with_match_info
from services.student_grading.bundler import _write_bundle_tex_inline_answers  # uses your existing bundler writer
from common.tex.compile_tex_to_pdf import compile_tex_to_pdf
from common.exam_summary import (
    _safe_str,
    compare_summaries,
    read_json_file,
    write_student_summary as write_student_summary_file,
)
from services.student_grading.unified_tex import (
    build_graded_result_json,
    build_graded_result_tex,
    build_student_feedback_json,
)
from core.ai_clients.ai_usage_logger import bind_usage_context
from services.student_work_store import (
    attach_grading_feedback_to_student_work,
    create_tex_student_work,
    resolve_student_work_file,
)
from services.student_access import get_student_code_record

router = APIRouter(prefix="/routes", tags=["grading"])

DEBUG = True


# -------------------------
# Small helpers (already in your style)
# -------------------------


def _response_for_path(
    p: Path,
    *,
    headers: dict[str, str] | None = None,
) -> FileResponse:
    if p.suffix.lower() == ".pdf":
        return FileResponse(
            path=str(p),
            media_type="application/pdf",
            filename="graded_test.pdf",
            headers=headers or {},
        )

    return FileResponse(
        path=str(p),
        media_type="text/plain; charset=utf-8",
        filename="graded_union.tex",
        headers=headers or {},
    )


def _require_provider_env(provider: str) -> None:
    p = (provider or "").strip().lower()
    if p in ("google", "gemini", "google_ai_studio", "aistudio"):
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise HTTPException(status_code=400, detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")

    if p in ("chatgpt", "openai", "gpt"):
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_key:
            raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment.")


def _safe_name(s: str, fallback: str = "unknown") -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _display_provider(provider: str) -> str:
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
    source_work_id: str | None = None,
    structured_json_path: Path | None = None,
    graded_tex_path: Path | None = None,
) -> None:
    meta = {
        "schema_version": (
            "grading_result_metadata_v2"
        ),
        "student_code": student_code,
        "exam_id": exam_id,
        "provider": provider,
        "debug": debug,
        "saved_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "final_file": final_path.name,
        "structured_json_file": (
            Path(
                structured_json_path
            ).name
            if structured_json_path
            else ""
        ),
        "graded_tex_file": (
            Path(
                graded_tex_path
            ).name
            if graded_tex_path
            else ""
        ),
        "source_work_id": (
            source_work_id or ""
        ),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _scan_complete_answer_objects(
        raw_text: str,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Recover complete answer objects from a possibly truncated JSON string.

    This supports OCR output such as:

      {
        "answers": [
          {...complete answer...},
          {...complete answer...},
          {...unfinished final answer...
        ]
      }

    Complete answer objects are preserved. The unfinished final object is
    skipped because its missing content cannot be reconstructed safely.
    """
    text = str(raw_text or "")

    marker = re.search(
        r'"answers"\s*:\s*\[',
        text,
        flags=re.IGNORECASE,
    )

    scan = text[marker.end():] if marker else text

    answers: list[dict[str, Any]] = []

    object_start = -1
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(scan):
        if in_string:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                object_start = index

            depth += 1
            continue

        if char != "}" or depth <= 0:
            continue

        depth -= 1

        if depth != 0 or object_start < 0:
            continue

        candidate = scan[
            object_start:index + 1
        ]

        object_start = -1

        if '"answer_text"' not in candidate:
            continue

        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(parsed, dict):
            continue

        if not str(
                parsed.get("answer_text") or ""
        ).strip():
            continue

        answers.append(parsed)

    # A positive depth means that the source ended while an object was open.
    return answers, depth > 0


def _embedded_answer_bundle(
        raw_text: str,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Parse a JSON answer bundle stored inside an answer_text field.

    First try ordinary JSON parsing. If the model response was truncated,
    recover every complete answer object separately.
    """
    text = str(raw_text or "").strip()

    if not text or '"answers"' not in text:
        return None, False

    try:
        parsed = json.loads(text)

        if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("answers"), list)
        ):
            valid_answers = [
                item
                for item in parsed["answers"]
                if (
                        isinstance(item, dict)
                        and str(
                    item.get("answer_text") or ""
                ).strip()
                )
            ]

            if valid_answers:
                parsed = dict(parsed)
                parsed["answers"] = valid_answers
                return parsed, False
    except Exception:
        pass

    recovered_answers, is_partial = (
        _scan_complete_answer_objects(text)
    )

    if not recovered_answers:
        return None, is_partial

    return {
        "schema_version": "student_answer_bundle_v1",
        "answers": recovered_answers,
    }, is_partial


def _flatten_embedded_answer_bundle(
    bundle: dict[str, Any],
    *,
    depth: int = 0,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    bool,
]:
    """
    Flatten answer bundles embedded inside answer_text.

    Supports both forms:

      1. A normal bundle:
         {"answers": [{question_id, part_key, answer_text}, ...]}

      2. A damaged wrapper:
         {
           "answers": [
             {
               "question_id": 1,
               "answer_text": "{\"answers\": [...]}"
             }
           ]
         }

    Recursion is limited so malformed model output cannot cause an
    infinite loop.
    """
    if depth > 5:
        return [], {}, False

    raw_answers = bundle.get("answers")

    if not isinstance(raw_answers, list):
        return [], {}, False

    flattened: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    source_was_partial = False

    for key in (
        "student_name",
        "student_id",
        "exam_id",
        "source_name",
        "document_type",
        "ocr_provider",
        "ocr_model",
    ):
        value = bundle.get(key)

        if value not in (None, ""):
            metadata[key] = value

    for answer in raw_answers:
        if not isinstance(answer, dict):
            continue

        answer_text = str(
            answer.get("answer_text") or ""
        ).strip()

        nested_bundle = None
        nested_partial = False

        if '"answers"' in answer_text:
            nested_bundle, nested_partial = (
                _embedded_answer_bundle(
                    answer_text
                )
            )

        source_was_partial = (
            source_was_partial
            or nested_partial
        )

        if nested_bundle:
            (
                nested_answers,
                nested_metadata,
                deeper_partial,
            ) = _flatten_embedded_answer_bundle(
                nested_bundle,
                depth=depth + 1,
            )

            source_was_partial = (
                source_was_partial
                or deeper_partial
            )

            if nested_answers:
                flattened.extend(
                    nested_answers
                )

                for key, value in (
                    nested_metadata.items()
                ):
                    if value not in (
                        None,
                        "",
                    ):
                        metadata[key] = value

                continue

        if answer_text:
            flattened.append(
                dict(answer)
            )

    return (
        flattened,
        metadata,
        source_was_partial,
    )


def _normalize_student_bundle_for_grading(
    input_path: Path,
    *,
    out_dir: Path,
) -> tuple[
    Path,
    dict[str, Any] | None,
]:
    """
    Recover structured OCR answers before grading.

    The structured bundle may arrive as:

      - a regular .json file;
      - JSON inside a .tex document body;
      - JSON inside one outer answer_text field;
      - a truncated JSON response containing several complete answers.

    The original uploaded file is never changed. When structured answers
    are recovered, a normalized JSON file is written beside the temporary
    grading artifacts.
    """
    input_path = Path(input_path)

    try:
        raw_text = input_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return input_path, None

    source_text = raw_text.strip()

    if not source_text:
        return input_path, None

    # OCR review sends a .tex file whose document body may actually be
    # the structured JSON returned by the OCR provider.
    begin_document = r"\begin{document}"
    end_document = r"\end{document}"

    if begin_document in source_text:
        source_text = source_text.split(
            begin_document,
            1,
        )[1]

    if end_document in source_text:
        source_text = source_text.rsplit(
            end_document,
            1,
        )[0]

    source_text = source_text.strip()

    # Remove wrappers sometimes used when JSON is placed in a TeX file.
    source_text = re.sub(
        r"^\s*\\begin\{(?:verbatim|lstlisting)\}\s*",
        "",
        source_text,
        flags=re.IGNORECASE,
    )

    source_text = re.sub(
        r"\s*\\end\{(?:verbatim|lstlisting)\}\s*$",
        "",
        source_text,
        flags=re.IGNORECASE,
    )

    if '"answers"' not in source_text:
        return input_path, None

    recovered_bundle, source_was_partial = (
        _embedded_answer_bundle(
            source_text
        )
    )

    if not recovered_bundle:
        return input_path, None

    (
        recovered_answers,
        recovered_metadata,
        nested_source_was_partial,
    ) = _flatten_embedded_answer_bundle(
        recovered_bundle
    )

    source_was_partial = (
        source_was_partial
        or nested_source_was_partial
    )

    if not recovered_answers:
        return input_path, None

    unique_answers: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, str, str]
    ] = set()

    for answer in recovered_answers:
        question_id = str(
            answer.get("question_id")
            or answer.get("questionId")
            or ""
        ).strip()

        part_key = str(
            answer.get("part_key")
            or answer.get("partKey")
            or answer.get("part")
            or ""
        ).strip()

        answer_text = str(
            answer.get("answer_text")
            or answer.get(
                "student_answer"
            )
            or ""
        ).strip()

        if (
            not question_id
            or not answer_text
        ):
            continue

        signature = (
            question_id,
            part_key.lower(),
            answer_text,
        )

        if signature in seen:
            continue

        seen.add(signature)

        normalized_answer = dict(
            answer
        )

        normalized_answer[
            "question_id"
        ] = question_id

        normalized_answer[
            "part_key"
        ] = part_key

        normalized_answer[
            "answer_text"
        ] = answer_text

        unique_answers.append(
            normalized_answer
        )

    if not unique_answers:
        return input_path, None

    normalized = {
        "schema_version": (
            "student_answer_bundle_v1"
        ),
        "student_name": str(
            recovered_metadata.get(
                "student_name"
            )
            or ""
        ),
        "student_id": str(
            recovered_metadata.get(
                "student_id"
            )
            or ""
        ),
        "exam_id": str(
            recovered_metadata.get(
                "exam_id"
            )
            or ""
        ),
        "source_name": str(
            recovered_metadata.get(
                "source_name"
            )
            or input_path.name
        ),
        "document_type": str(
            recovered_metadata.get(
                "document_type"
            )
            or "handwritten_scan"
        ),
        "ocr_provider": str(
            recovered_metadata.get(
                "ocr_provider"
            )
            or ""
        ),
        "ocr_model": (
            recovered_metadata.get(
                "ocr_model"
            )
        ),
        "answers": unique_answers,
        "normalization": {
            "applied": True,
            "source_filename": (
                input_path.name
            ),
            "source_suffix": (
                input_path.suffix.lower()
            ),
            "reason": (
                "Recovered structured OCR "
                "answers from the uploaded "
                "grading file."
            ),
            "recovered_answer_count": (
                len(unique_answers)
            ),
            "source_was_partial": (
                source_was_partial
            ),
        },
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_path = (
        out_dir
        / "student_answer_bundle_normalized.json"
    )

    normalized_path.write_text(
        json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        normalized_path,
        normalized["normalization"],
    )
# -------------------------
# Main grading flow (single extraction, single source-of-truth payloads)
# -------------------------

async def _grade_tex_flow(
    *,
    provider: str,
    student_tex: UploadFile,
    job_id: str | None,
    session: dict,
    request: Request | None,
    selected_exam_id: str | None = None,
    source_work_id: str | None = None,
    debug: bool = False,
) -> FileResponse:
    # Trust only the session for identity; never the form field.
    student_code = session["sub"] if session.get("role") == "student" else None
    source_work_id = (source_work_id or "").strip() if student_code else ""
    """
    Canonical flow (no duplicate extraction):

      1) Save student upload
      2) Match reference in solution bank
      3) Build payloads ONCE (full_solution_bundle.json + student snippets when available)
      4) Write Q/A bundle TeX from those payloads (display/export only)
      5) Grade each payload (AI) -> grades.json (+ usage_summary.json)
      6) Build feedback TeX from grades.json
      7) Unify bundle + feedback into one TeX
      8) Compile unified TeX to PDF (fallback to TeX)
      9) Persist final output and return it
    """
    _require_provider_env(provider)
    normalized_provider = _display_provider(provider)

    trace = create_debug_trace(
        "grading",
        provider=normalized_provider,
        job_id=job_id,
        student_filename=student_tex.filename,
        session_role=session.get("role") if session else None,
        student_code=student_code,
    )
    session_role = session.get("role")
    teacher_id = (session.get("teacher_id") or "").strip()

    if session_role == "teacher" and not teacher_id:
        teacher_id = (session.get("sub") or "").strip()

    if session_role == "student" and not teacher_id:
        raise HTTPException(
            status_code=403,
            detail="Student code is not assigned to a teacher/course.",
        )

    bank_voucher_id = ""

    if session_role == "student" and student_code:
        try:
            student_record = get_student_code_record(student_code) or {}
            bank_voucher_id = str(
                student_record.get("voucher_id")
                or student_record.get("voucher_hash")
                or student_record.get("voucher_code")
                or ""
            ).strip()
        except Exception:
            bank_voucher_id = ""

    active_bank_root = teacher_bank_root(teacher_id, voucher_id=bank_voucher_id) if teacher_id else BANK_ROOT
    bind_bank_root(active_bank_root)

    bind_usage_context(
        debug_run_id=trace.run_id,
        route="grade_tex",
        provider=normalized_provider,
        session_role=session_role,
        student_code=student_code or None,
        teacher_id=teacher_id or None,
        voucher_id=bank_voucher_id or None,
    )
    trace.log("grading", "started", provider=normalized_provider, student_filename=student_tex.filename, selected_exam_id=selected_exam_id if session.get("role") == "teacher" else None)

    persistent_results_dir = RUNS_ROOT / "final_results"
    persistent_results_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir: Path | None = None
    student_summary_path: Path | None = None

    try:
        if job_id:
            init_job(job_id)

        # Stage 1: temp workspace
        tmp_dir = Path(tempfile.mkdtemp(prefix="mathgrade_", dir=str(RUNS_ROOT)))
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2: save student
        uploaded_name = Path(student_tex.filename or "student.tex").name
        tex_path = tmp_dir / uploaded_name
        student_upload_bytes = await student_tex.read()
        tex_path.write_bytes(student_upload_bytes)
        artifact_name = "student_answer_bundle.json" if tex_path.suffix.lower() == ".json" else f"student_upload{tex_path.suffix or '.tex'}"
        trace.save_bytes(
            artifact_name,
            student_upload_bytes,
            stage="student_upload",
        )

        trace.log(
            "student_upload",
            "saved",
            path=str(tex_path),
            bytes=len(student_upload_bytes),
            suffix=tex_path.suffix.lower(),
        )

        # Some OCR providers occasionally return a JSON answer bundle as
        # text inside one outer answer_text field. Recover the inner answer
        # objects before reference matching and payload construction.
        grading_input_path, normalization_info = (
            _normalize_student_bundle_for_grading(
                tex_path,
                out_dir=out_dir,
            )
        )

        if grading_input_path != tex_path:
            trace.save_file(
                grading_input_path,
                "student_answer_bundle_normalized.json",
                stage="student_upload",
            )

            trace.log(
                "student_upload",
                "structured_bundle_normalized",
                **(normalization_info or {}),
            )

            if job_id:
                recovered_count = int(
                    (
                        normalization_info
                        or {}
                    ).get(
                        "recovered_answer_count"
                    )
                    or 0
                )

                push(
                    job_id,
                    (
                        f"Recovered {recovered_count} "
                        "structured OCR answer parts"
                    ),
                )

        # If this is a direct student TeX/TXT upload, create a canonical
        # student-work folder now. OCR uploads already created one earlier and
        # pass source_work_id from the frontend.
        if student_code and not source_work_id:
            try:
                created_work = create_tex_student_work(
                    student_code=student_code,
                    teacher_id=teacher_id,
                    source_filename=uploaded_name,
                    uploaded_bytes=student_upload_bytes,
                    debug_trace_id=trace.run_id,
                    debug_trace_dir=str(trace.path) if trace.enabled else "",
                )
                source_work_id = str(created_work.get("work_id") or "").strip()
                trace.log(
                    "student_work_store",
                    "direct_tex_saved",
                    work_id=source_work_id,
                    teacher_id=created_work.get("teacher_id"),
                    voucher_id=created_work.get("voucher_id"),
                )
            except Exception as e:
                trace.log(
                    "student_work_store",
                    "direct_tex_save_failed",
                    status="warning",
                    error=_safe_str(e, 500),
                )
        if job_id:
            push(job_id, "Saved student file")

        # summary (optional)
        try:
            summary_path = write_student_summary_file(
                                                        grading_input_path,
                                                        out_dir=out_dir,
                                                    )
            student_summary_path = summary_path
            trace.save_file(summary_path, "student_summary.json", stage="student_summary")
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                trace.log(
                    "student_summary",
                    "built",
                    parse_ok=bool(summary_data.get("parse_ok")),
                    qnums=summary_data.get("qnums") or [],
                    keys_preview=summary_data.get("keys_preview") or [],
                    preview_chars=len(summary_data.get("preview_text") or ""),
                )
            except Exception as summary_log_error:
                trace.log(
                    "student_summary",
                    "log_summary_failed",
                    status="warning",
                    error=_safe_str(summary_log_error, 300),
                )
            if job_id:
                push(job_id, f"Generated student summary: {summary_path.name}")
        except Exception as e:
            trace.log("student_summary", "failed", status="warning", error=_safe_str(e, 500))
            if job_id:
                push(job_id, f"Student summary failed (continuing): {_safe_str(e, 220)}")

        # Stage 3: match reference
        if job_id:
            push(job_id, "Fetching reference from solution bank…")

        if not active_bank_root.exists():
            raise RuntimeError(f"Solution bank folder does not exist: {active_bank_root}")

        t_ref = perf_counter()
        manual_exam_id = (selected_exam_id or "").strip() if session.get("role") == "teacher" else ""

        if manual_exam_id:
            exam_id = require_safe_exam_id(manual_exam_id)
            uploads = active_bank_root / exam_id / "uploads"
            json_ref = uploads / "full_solution_bundle.json"
            tex_ref = uploads / "reference_current.tex"
            if json_ref.exists():
                chosen_ref = json_ref
            elif tex_ref.exists():
                chosen_ref = tex_ref
            else:
                raise HTTPException(status_code=404, detail=f"Selected reference exam has no full_solution_bundle.json or reference_current.tex: {exam_id}")

            try:
                write_reference_summary(exam_id)
            except Exception as e:
                trace.log("reference_match", "manual_summary_refresh_failed", status="warning", error=_safe_str(e, 500))

            match_method = "manual_teacher_selection"
            match_reason = "Teacher test mode selected the reference explicitly."
            match_score = None
            student_qnums = []
            reference_qnums = []
            reference_summary_path = uploads / "reference_summary.json"
            try:
                if student_summary_path and student_summary_path.exists():
                    student_summary_for_log = read_json_file(student_summary_path) or {}
                    student_qnums = [int(x) for x in (student_summary_for_log.get("qnums") or []) if str(x).isdigit()]
                reference_summary_for_log = read_json_file(reference_summary_path) or {}
                reference_qnums = [int(x) for x in (reference_summary_for_log.get("qnums") or []) if str(x).isdigit()]
            except Exception:
                pass
        else:
            match = await run_in_threadpool(
                pick_reference_with_match_info,
                bank_dir=active_bank_root,
                student_tex=grading_input_path,
                prefer_heuristic=True,
                llm_top_k=12,
            )
            exam_id, chosen_ref = match.exam_id, match.path
            match_method = match.method
            match_reason = match.reason
            match_score = match.score
            student_qnums = list(match.student_qnums)
            reference_qnums = list(match.reference_qnums)

        ref_secs = perf_counter() - t_ref
        reference_source = "json" if Path(chosen_ref).suffix.lower() == ".json" else "tex_fallback"
        reference_selection_mode = "manual" if manual_exam_id else "automatic"

        # The exam is only known after reference matching. Extend the existing
        # request-level usage context before any grading model calls are made.
        bind_usage_context(
            exam_id=str(exam_id),
            content_type="student_grading",
            reference_selection_mode=reference_selection_mode,
        )

        trace.log(
            "reference_match",
            "selected",
            exam_id=str(exam_id),
            reference_path=str(chosen_ref),
            reference_source=reference_source,
            reference_selection_mode=reference_selection_mode,
            match_method=match_method,
            match_reason=match_reason,
            match_score=match_score,
            student_qnums=student_qnums,
            reference_qnums=reference_qnums,
            duration_s=round(ref_secs, 3),
        )
        if job_id:
            source_label = "JSON reference" if reference_source == "json" else "TeX fallback reference"
            mode_label = "manual" if manual_exam_id else "auto"
            push(job_id, f"Fetched: {exam_id} ({source_label}, {mode_label} reference, {ref_secs:.1f}s)")

        ref_path = (
                tmp_dir
                / Path(chosen_ref).name
        )

        ref_path.write_text(
            Path(chosen_ref).read_text(
                encoding="utf-8",
                errors="replace",
            ),
            encoding="utf-8",
        )

        # A generated full_solution_bundle.json can lose formulas that
        # originally appeared in TeX subsection titles. Copy its source
        # TeX files into the temporary workspace so payloads.py can restore
        # those titles deterministically.
        if reference_source == "json":
            try:
                reference_bundle_data = (
                    json.loads(
                        Path(
                            chosen_ref
                        ).read_text(
                            encoding="utf-8",
                        )
                    )
                )

                source_reference_names: set[
                    str
                ] = set()

                source_names = (
                    reference_bundle_data.get(
                        "source_names"
                    )
                    if isinstance(
                        reference_bundle_data,
                        dict,
                    )
                    else []
                )

                if isinstance(
                        source_names,
                        list,
                ):
                    for source_name in source_names:
                        safe_source_name = (
                            Path(
                                str(
                                    source_name
                                    or ""
                                )
                            ).name
                        )

                        if safe_source_name:
                            source_reference_names.add(
                                safe_source_name
                            )

                questions = (
                    reference_bundle_data.get(
                        "questions"
                    )
                    if isinstance(
                        reference_bundle_data,
                        dict,
                    )
                    else []
                )

                if isinstance(
                        questions,
                        list,
                ):
                    for question in questions:
                        if not isinstance(
                                question,
                                dict,
                        ):
                            continue

                        parts = (
                            question.get(
                                "parts"
                            )
                            if isinstance(
                                question.get(
                                    "parts"
                                ),
                                list,
                            )
                            else []
                        )

                        for part in parts:
                            if not isinstance(
                                    part,
                                    dict,
                            ):
                                continue

                            for field_name in (
                                    "source_question_file",
                                    "source_answer_file",
                            ):
                                safe_source_name = (
                                    Path(
                                        str(
                                            part.get(
                                                field_name
                                            )
                                            or ""
                                        )
                                    ).name
                                )

                                if safe_source_name:
                                    source_reference_names.add(
                                        safe_source_name
                                    )

                source_parent = (
                    Path(chosen_ref).parent
                )

                copied_source_count = 0

                for source_name in sorted(
                        source_reference_names
                ):
                    candidate_paths: list[
                        Path
                    ] = []

                    # Prefer a source file stored directly beside the
                    # selected reference bundle.
                    direct_candidate = (
                            source_parent
                            / source_name
                    )

                    if (
                            direct_candidate.exists()
                            and direct_candidate.is_file()
                    ):
                        candidate_paths.append(
                            direct_candidate
                        )

                    # Generated JSON and source TeX may be stored in
                    # different subfolders of the same solution bank.
                    try:
                        bank_root = Path(
                            active_bank_root
                        )

                        if bank_root.exists():
                            for candidate in (
                                    bank_root.rglob(
                                        source_name
                                    )
                            ):
                                if candidate.is_file():
                                    candidate_paths.append(
                                        candidate
                                    )
                    except Exception as search_error:
                        trace.log(
                            "reference_match",
                            "reference_source_search_failed",
                            status="warning",
                            source_name=source_name,
                            error=_safe_str(
                                search_error,
                                500,
                            ),
                        )

                    # Remove duplicate paths while preserving priority.
                    unique_candidates: list[
                        Path
                    ] = []

                    seen_candidate_paths: set[
                        str
                    ] = set()

                    for candidate in (
                            candidate_paths
                    ):
                        try:
                            identity = str(
                                candidate.resolve()
                            )
                        except Exception:
                            identity = str(
                                candidate
                            )

                        if (
                                identity
                                in seen_candidate_paths
                        ):
                            continue

                        seen_candidate_paths.add(
                            identity
                        )

                        unique_candidates.append(
                            candidate
                        )

                    if not unique_candidates:
                        trace.log(
                            "reference_match",
                            "reference_source_file_missing",
                            status="warning",
                            source_name=source_name,
                            searched_reference_parent=str(
                                source_parent
                            ),
                            searched_bank_root=str(
                                active_bank_root
                            ),
                        )
                        continue

                    source_path = (
                        unique_candidates[0]
                    )

                    if (
                            len(unique_candidates)
                            > 1
                    ):
                        trace.log(
                            "reference_match",
                            "multiple_reference_source_files_found",
                            status="warning",
                            source_name=source_name,
                            selected_path=str(
                                source_path
                            ),
                            candidate_paths=[
                                str(candidate)
                                for candidate
                                in unique_candidates
                            ],
                        )

                    copied_source_path = (
                            tmp_dir
                            / source_name
                    )

                    shutil.copy2(
                        source_path,
                        copied_source_path,
                    )

                    copied_source_count += 1

                    trace.save_file(
                        copied_source_path,
                        (
                            "reference_sources/"
                            f"{source_name}"
                        ),
                        stage="reference_match",
                    )

                    trace.log(
                        "reference_match",
                        "reference_source_file_copied",
                        source_name=source_name,
                        copied_path=str(
                            copied_source_path
                        ),
                    )

                    trace.log(
                        "reference_match",
                        "reference_source_copy_summary",
                        requested_count=len(
                            source_reference_names
                        ),
                        copied_count=(
                            copied_source_count
                        ),
                        temporary_directory=str(
                            tmp_dir
                        ),
                    )

                    if (
                            source_reference_names
                            and copied_source_count == 0
                    ):
                        raise RuntimeError(
                            "The full solution JSON refers "
                            "to source TeX files, but none "
                            "could be located in the active "
                            "solution bank. Cannot restore "
                            "complete question formulas."
                        )

            except Exception as source_copy_error:
                trace.log(
                    "reference_match",
                    "reference_source_copy_failed",
                    status="warning",
                    error=_safe_str(
                        source_copy_error,
                        500,
                    ),
                )

        matched_artifact_name = (
            "matched_reference.json"
            if reference_source == "json"
            else "matched_reference.tex"
        )
        trace.save_file(ref_path, matched_artifact_name, stage="reference_match")

        # Compare student_summary.json to reference_summary.json before building payloads.
        # This makes matching failures and OCR structure problems visible even when
        # we use the single-exam fallback.
        try:
            if student_summary_path is None or not student_summary_path.exists():
                student_summary_path = write_student_summary_file(
                                        grading_input_path,
                                        out_dir=out_dir,
                                    )

            reference_summary_path = active_bank_root / str(exam_id) / "uploads" / "reference_summary.json"
            student_summary = read_json_file(student_summary_path) or {}
            reference_summary = read_json_file(reference_summary_path) or {}

            summary_compare = compare_summaries(
                student_summary=student_summary,
                reference_summary=reference_summary,
                exam_id=str(exam_id),
                reference_path=str(chosen_ref),
                match_method=match_method,
                match_reason=match_reason,
            )
            summary_compare_path = out_dir / "summary_compare.json"
            summary_compare_path.write_text(
                json.dumps(summary_compare, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            trace.save_file(summary_compare_path, "summary_compare.json", stage="reference_match")
            trace.log(
                "reference_match",
                "summary_compared",
                status=summary_compare.get("status", "ok"),
                qnum_overlap=summary_compare.get("overlap", {}).get("qnum_overlap", []),
                qnum_overlap_count=summary_compare.get("overlap", {}).get("qnum_overlap_count", 0),
                part_overlap_count=summary_compare.get("overlap", {}).get("part_overlap_count", 0),
                warnings=summary_compare.get("warnings", []),
            )
        except Exception as e:
            trace.log(
                "reference_match",
                "summary_compare_failed",
                status="warning",
                error=_safe_str(e, 500),
            )

        if grading_input_path != tex_path:
            try:
                normalized_check = json.loads(
                    grading_input_path.read_text(
                        encoding="utf-8"
                    )
                )

                normalized_answers = (
                    normalized_check.get(
                        "answers"
                    )
                    if isinstance(
                        normalized_check,
                        dict,
                    )
                    else []
                )

                normalized_answer_count = (
                    len(normalized_answers)
                    if isinstance(
                        normalized_answers,
                        list,
                    )
                    else 0
                )

                trace.log(
                    "student_upload",
                    "normalized_bundle_ready_for_payloads",
                    normalized_path=str(
                        grading_input_path
                    ),
                    answer_count=(
                        normalized_answer_count
                    ),
                )

                if (
                    normalized_answer_count
                    == 0
                ):
                    raise RuntimeError(
                        "The normalized OCR bundle "
                        "contains no answers."
                    )

            except Exception as normalization_error:
                raise RuntimeError(
                    "Structured OCR normalization "
                    "failed before payload creation: "
                    f"{normalization_error}"
                ) from normalization_error
        # Stage 4: build payloads ONCE (source-of-truth)
        if job_id:
            push(job_id, "Extracting parts (building payloads)…")

        payload_out = out_dir / "payloads_run"
        payload_out.mkdir(parents=True, exist_ok=True)

        t_payloads = perf_counter()
        manifest_json, items = await run_in_threadpool(
            build_payloads,
            reference_tex=ref_path,
            student_tex=grading_input_path,
            out_dir=payload_out,
            default_max_points=0.0,
        )
        payload_secs = perf_counter() - t_payloads
        trace.save_file(manifest_json, "payload_manifest.json", stage="payloads")
        trace.log(
            "payloads",
            "built",
            duration_s=round(payload_secs, 3),
            payload_count=len(items),
            reference_source=reference_source,
        )
        for it in items[:80]:
            trace.save_file(it.payload_path, f"payloads/{it.payload_path.name}", stage="payloads")

        if job_id:
            push(job_id, f"Payloads ready ({payload_secs:.1f}s)")

        # Stage 4b: write bundle TeX from the SAME payloads (no re-extraction)
        if job_id:
            push(job_id, "Generating Q/A bundle (TeX)…")

        reference_snippets: Dict[Tuple[int, str | None], str] = {}
        student_answers: Dict[Tuple[int, str | None], str] = {}

        # We reuse the payload json content; keep keys aligned with bundler expectations.
        for it in items:
            data = json.loads(it.payload_path.read_text(encoding="utf-8"))
            key_dict = data.get("key") or {}
            qnum = int(key_dict.get("qnum"))
            part = key_dict.get("part") or ""
            key = (qnum, part)

            q_text = (data.get("reference") or {}).get("question_text") or ""
            sol_text = (data.get("reference") or {}).get("solution_text") or ""
            ref_block = "\n\n".join([x for x in [q_text.strip(), sol_text.strip()] if x.strip()]).strip()
            reference_snippets[key] = ref_block

            stu_raw = (data.get("student") or {}).get("latex_raw") or ""
            student_answers[key] = stu_raw.strip()

        bundle_tex = out_dir / "qa_bundle.tex"
        await run_in_threadpool(
            _write_bundle_tex_inline_answers,
            bundle_tex,
            title="Q/A Bundle (Reference + Student Answer)",
            font_name=FIXED_FONT,
            reference_snippets=reference_snippets,
            student_answers=student_answers,
        )
        trace.save_file(bundle_tex, "qa_bundle.tex", stage="qa_bundle")
        if job_id:
            push(job_id, "Q/A bundle TeX ready")

        # Stage 5: grade each payload (AI) using the SAME manifest
        ai_dir = out_dir / "ai_grade"
        ai_dir.mkdir(parents=True, exist_ok=True)

        if job_id:
            push(job_id, "Grading please wait…")

        t_grade = perf_counter()
        grades_json = await run_in_threadpool(
            grade_payload_manifest,
            manifest_json=manifest_json,
            out_dir=ai_dir,
            model=provider,
            debug=debug,
            log_fn=(lambda msg: push(job_id, msg)) if job_id else None,
        )
        grade_secs = perf_counter() - t_grade
        trace.save_file(grades_json, "grades.json", stage="grading")
        trace.log("grading", "completed", duration_s=round(grade_secs, 3), grades_path=str(grades_json))
        if job_id:
            push(job_id, f"Grading finished ({grade_secs:.1f}s)")

        # Stage 6: read usage and log submission (students only; admins are skipped in logger)
        gemini_tokens = 0
        try:
            usage_path = ai_dir / "usage_summary.json"
            if usage_path.exists():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
                trace.save_file(usage_path, "usage_summary.json", stage="usage")
                if (usage.get("provider") or "").lower() in ("google", "gemini", "google_ai_studio", "aistudio"):
                    gemini_tokens = int(usage.get("total_tokens") or 0)
        except Exception:
            gemini_tokens = 0

        try:
            session_role = session.get("role") if session else None

            # Only real student submissions are logged.
            # Teacher test-mode submissions are graded, but skipped here.
            if session_role == "student" and student_code:
                ua = request.headers.get("user-agent", "") if request else ""
                ip = request.client.host if (request and request.client) else ""

                log_student_submission(
                    code=student_code,
                    exam_id=str(exam_id),
                    provider=provider,
                    ip=ip,
                    user_agent=ua,
                    gemini_tokens=gemini_tokens,
                    session_role=session_role,
                )
        except Exception:
            pass

        # Stage 7: canonical structured graded result
        if job_id:
            push(
                job_id,
                "Building structured graded result…",
            )

        t_structured = perf_counter()

        graded_result_json = (
            await run_in_threadpool(
                build_graded_result_json,
                manifest_json=manifest_json,
                grades_json=grades_json,
                out_dir=ai_dir,
                exam_id=str(exam_id),
                provider=normalized_provider,
                student_code=student_code or "",
                source_filename=uploaded_name,
            )
        )

        overlay_bundle_json = (
            grading_input_path
            if grading_input_path.suffix.lower()
            == ".json"
            else None
        )

        # Prefer the original saved OCR bundle for page coordinates. This keeps
        # the visual anchors even when the student edits the reviewable LaTeX
        # before grading.
        if student_code and source_work_id:
            try:
                stored_overlay_bundle = (
                    resolve_student_work_file(
                        student_code,
                        source_work_id,
                        "student_answer_bundle.json",
                    )
                )

                if stored_overlay_bundle.exists():
                    overlay_bundle_json = (
                        stored_overlay_bundle
                    )
            except HTTPException:
                pass

        student_feedback_json = (
            await run_in_threadpool(
                build_student_feedback_json,
                graded_result_json=(
                    graded_result_json
                ),
                student_answer_bundle_json=(
                    overlay_bundle_json
                ),
                out_dir=ai_dir,
                output_name=(
                    "student_feedback.json"
                ),
            )
        )

        structured_secs = (
                perf_counter()
                - t_structured
        )

        trace.save_file(
            graded_result_json,
            "graded_result_internal.json",
            stage="graded_result",
        )

        trace.save_file(
            student_feedback_json,
            "student_feedback.json",
            stage="graded_result",
        )

        try:
            structured_data = json.loads(
                graded_result_json.read_text(
                    encoding="utf-8"
                )
            )

            structured_part_count = len(
                structured_data.get(
                    "parts"
                )
                or []
            )
        except Exception:
            structured_part_count = 0

        trace.log(
            "graded_result",
            "built",
            duration_s=round(
                structured_secs,
                3,
            ),
            path=str(
                graded_result_json
            ),
            part_count=(
                structured_part_count
            ),
        )

        if job_id:
            push(
                job_id,
                (
                    "Structured graded result "
                    f"ready with "
                    f"{structured_part_count} "
                    f"parts "
                    f"({structured_secs:.1f}s)"
                ),
            )

        # Stage 8: render the final tandem TeX from the JSON
        if job_id:
            push(
                job_id,
                (
                    "Generating Question → "
                    "Student answer → Feedback "
                    "document…"
                ),
            )

        t_tex = perf_counter()

        final_tex = await run_in_threadpool(
            build_graded_result_tex,
            graded_result_json=(
                graded_result_json
            ),
            out_dir=ai_dir,
            output_stem=(
                f"graded_{normalized_provider}"
            ),
            font_name=FIXED_FONT,
        )

        tex_secs = (
                perf_counter()
                - t_tex
        )

        trace.save_file(
            final_tex,
            "final_structured.tex",
            stage="graded_result_tex",
        )

        trace.log(
            "graded_result_tex",
            "built",
            duration_s=round(
                tex_secs,
                3,
            ),
            path=str(final_tex),
        )

        if job_id:
            push(
                job_id,
                (
                    "Structured grading TeX "
                    f"ready ({tex_secs:.1f}s)"
                ),
            )

        # Stage 9: compile final TeX to PDF (fallback to TeX)
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
                    texinputs=[
                        final_tex.parent,
                    ],
                ).pdf
            )
            compiled_pdf = Path(compiled_pdf)
            pdf_secs = perf_counter() - t_pdf

            if compiled_pdf.exists() and compiled_pdf.suffix.lower() == ".pdf":
                result_path = compiled_pdf
                trace.save_file(compiled_pdf, "final_output.pdf", stage="pdf_compile")
                trace.log("pdf_compile", "completed", duration_s=round(pdf_secs, 3), path=str(compiled_pdf))
                if job_id:
                    push(job_id, f"PDF build finished ({pdf_secs:.1f}s)")
            else:
                result_path = final_tex
                trace.log("pdf_compile", "missing_pdf", status="warning", duration_s=round(pdf_secs, 3))
                if job_id:
                    push(job_id, f"PDF build did not produce a valid PDF ({pdf_secs:.1f}s) — falling back to TeX")
        except Exception as e:
            pdf_secs = perf_counter() - t_pdf
            result_path = final_tex
            trace.error("pdf_compile", e, duration_s=round(pdf_secs, 3))
            if job_id:
                push(job_id, f"PDF compile failed ({pdf_secs:.1f}s); falling back to TeX: {_safe_str(e, 220)}")

        # Stage 10: persist final output
        final_persistent_path = _build_persistent_result_path(
            persistent_root=persistent_results_dir,
            student_code=student_code,
            exam_id=str(exam_id),
            provider=normalized_provider,
            source_result_path=result_path,
        )
        shutil.copy2(
            result_path,
            final_persistent_path,
        )

        persistent_stem = (
            final_persistent_path.stem
        )

        internal_persistent_path = (
                final_persistent_path.parent
                / (
                    f"{persistent_stem}"
                    "_internal.json"
                )
        )

        structured_persistent_path = (
                final_persistent_path.parent
                / (
                    f"{persistent_stem}"
                    "_structured.json"
                )
        )

        shutil.copy2(
            graded_result_json,
            internal_persistent_path,
        )

        # Only this safe file is attached to the student's saved work.
        shutil.copy2(
            student_feedback_json,
            structured_persistent_path,
        )

        if (
            final_persistent_path.suffix.lower()
            == ".tex"
        ):
            graded_tex_persistent_path = (
                final_persistent_path
            )
        else:
            graded_tex_persistent_path = (
                final_persistent_path.parent
                / (
                    f"{persistent_stem}"
                    "_source.tex"
                )
            )

            shutil.copy2(
                final_tex,
                graded_tex_persistent_path,
            )

        persistent_metadata_path = (
            final_persistent_path.parent
            / (
                f"{persistent_stem}"
                "_metadata.json"
            )
        )

        result_saved_at = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        _write_result_metadata(
            meta_path=(
                persistent_metadata_path
            ),
            student_code=student_code,
            exam_id=str(exam_id),
            provider=normalized_provider,
            final_path=(
                final_persistent_path
            ),
            debug=debug,
            source_work_id=(
                source_work_id
            ),
            structured_json_path=(
                structured_persistent_path
            ),
            graded_tex_path=(
                graded_tex_persistent_path
            ),
        )

        response_headers: dict[str, str] = {}

        if student_code and source_work_id:
            try:
                updated_work = (
                    attach_grading_feedback_to_student_work(
                        student_code=(
                            student_code
                        ),
                        work_id=(
                            source_work_id
                        ),
                        exam_id=str(
                            exam_id
                        ),
                        provider=(
                            normalized_provider
                        ),
                        final_path=(
                            final_persistent_path
                        ),
                        graded_result_json_path=(
                            structured_persistent_path
                        ),
                        graded_result_tex_path=(
                            graded_tex_persistent_path
                        ),
                        saved_at=(
                            result_saved_at
                        ),
                    )
                )
                if updated_work:
                    trace.log(
                        "student_work_store",
                        "feedback_attached",
                        work_id=source_work_id,
                        exam_id=str(exam_id),
                        provider=normalized_provider,
                    )

                    feedback_meta = (
                        updated_work.get("feedback")
                        if isinstance(
                            updated_work.get("feedback"),
                            dict,
                        )
                        else {}
                    )

                    graded_result_meta = (
                        updated_work.get("graded_result")
                        if isinstance(
                            updated_work.get("graded_result"),
                            dict,
                        )
                        else {}
                    )

                    structured_filename = str(
                        feedback_meta.get(
                            "structured_json_filename"
                        )
                        or graded_result_meta.get(
                            "structured_json_filename"
                        )
                        or ""
                    ).strip()

                    if not structured_filename:
                        files = (
                            updated_work.get("files")
                            if isinstance(
                                updated_work.get("files"),
                                list,
                            )
                            else []
                        )

                        structured_entry = next(
                            (
                                item
                                for item in files
                                if isinstance(item, dict)
                                   and item.get("kind")
                                   == "graded_result_json"
                            ),
                            None,
                        )

                        if structured_entry:
                            structured_filename = str(
                                structured_entry.get(
                                    "filename"
                                )
                                or ""
                            ).strip()

                    if structured_filename:
                        structured_url = (
                            "/routes/student/work_file"
                            f"?work_id={quote(source_work_id)}"
                            f"&filename={quote(structured_filename)}"
                        )

                        response_headers[
                            "X-MathGrade-Structured-Url"
                        ] = structured_url

                        response_headers[
                            "X-MathGrade-Structured-Filename"
                        ] = structured_filename

                    response_headers[
                        "X-MathGrade-Work-Id"
                    ] = source_work_id
            except Exception as e:
                trace.log(
                    "student_work_store",
                    "feedback_attach_failed",
                    status="warning",
                    work_id=source_work_id,
                    error=_safe_str(e, 500),
                )
        trace.save_file(
            final_persistent_path,
            (
                "persisted_result"
                f"{final_persistent_path.suffix}"
            ),
            stage="persist",
        )

        trace.save_file(
            internal_persistent_path,
            "persisted_graded_result_internal.json",
            stage="persist",
        )

        trace.save_file(
            structured_persistent_path,
            "persisted_student_feedback.json",
            stage="persist",
        )

        trace.save_file(
            graded_tex_persistent_path,
            "persisted_graded_result.tex",
            stage="persist",
        )

        trace.save_file(
            persistent_metadata_path,
            "persisted_result_metadata.json",
            stage="persist",
        )
        trace.log(
            "grading",
            "finished",
            exam_id=str(exam_id),
            reference_source=reference_source,
            reference_selection_mode=reference_selection_mode,
            final_path=str(final_persistent_path),
        )

        if job_id:
            push(job_id, f"Done. Sending file: {final_persistent_path.name}")
            done(job_id)

        return _response_for_path(
            final_persistent_path,
            headers=response_headers,
        )

    except HTTPException as e:
        trace.error("http_exception", e)
        if job_id:
            fail(job_id, "HTTPException")
        raise

    except Exception as e:
        trace.error("grading", e)
        log_path = write_debug_log(f"grade_tex_{_display_provider(provider)}", e)
        if job_id:
            push(job_id, f"FAILED: {_safe_str(e, 400)}")
            fail(job_id, f"{e}")
        raise HTTPException(status_code=500, detail=f"{e}\n\nSaved traceback to: {log_path}")

    finally:
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
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    session: dict = Depends(require_session),
    source_work_id: str | None = Form(None),
):
    return await _grade_tex_flow(
        provider="ollama",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        selected_exam_id=selected_exam_id,
        source_work_id=source_work_id,
        debug=DEBUG,
    )


@router.post("/grade_tex_google")
async def grade_tex_google(
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    source_work_id: str | None = Form(None),
    session: dict = Depends(require_session),
):
    # IMPORTANT: pass "google" (not "gemini") so client selection + schema pruning match
    return await _grade_tex_flow(
        provider="google",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        selected_exam_id=selected_exam_id,
        source_work_id=source_work_id,
        debug=DEBUG,
    )


@router.post("/grade_tex_chatgpt")
async def grade_tex_chatgpt(
    request: Request,
    student_tex: UploadFile = File(...),
    job_id: str | None = Form(None),
    selected_exam_id: str | None = Form(None),
    source_work_id: str | None = Form(None),
    session: dict = Depends(require_session),
):
    return await _grade_tex_flow(
        provider="chatgpt",
        student_tex=student_tex,
        job_id=job_id,
        session=session,
        request=request,
        source_work_id=source_work_id,
        debug=DEBUG,
    )