from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from common.tex.part_normalize import normalize_part
from common.tex.student_tex import parse_student_tex_answers

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = _CONTROL_CHARS_RE.sub("", text.strip())
    # Strip common markdown fences without assuming they exist.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _clean_part(value: Any) -> str:
    return normalize_part(str(value or "").strip())


def _clean_answer_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Remove accidental feedback labels if a vision model over-helped.
    text = re.sub(r"(?im)^\s*(משוב|Feedback)\s*:\s*.*$", "", text).strip()
    return text


def normalize_student_answer_bundle(
    data: dict[str, Any] | None,
    *,
    source_name: str = "",
    provider: str = "",
    model: str | None = None,
    document_type: str = "unknown",
) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    raw_answers = data.get("answers") or []
    answers: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not isinstance(raw_answers, list):
        raw_answers = []
        warnings.append("Model output did not contain an answers list.")

    seen: set[tuple[int, str]] = set()
    for item in raw_answers:
        if not isinstance(item, dict):
            continue
        qnum = _as_int(item.get("question_id") or item.get("qnum") or item.get("question_number"))
        if qnum is None:
            warnings.append("Skipped an answer block with no question_id.")
            continue
        part = _clean_part(item.get("part_key") or item.get("part") or "")
        answer_text = _clean_answer_text(
            item.get("answer_text")
            or item.get("student_answer")
            or item.get("text")
            or item.get("raw_text")
            or ""
        )
        if not answer_text:
            warnings.append(f"Skipped Q{qnum}{part or ''}: empty answer_text.")
            continue

        key = (qnum, part)
        if key in seen:
            # Merge duplicate fragments conservatively.
            for existing in answers:
                if existing["question_id"] == qnum and existing.get("part_key", "") == part:
                    existing["answer_text"] = (existing["answer_text"].rstrip() + "\n\n" + answer_text).strip()
                    existing.setdefault("warnings", []).append("Merged duplicate OCR block for this question/part.")
                    break
            continue
        seen.add(key)

        page_numbers = item.get("page_numbers") or item.get("pages") or []
        if not isinstance(page_numbers, list):
            page_numbers = []
        normalized_pages: list[int] = []
        for p in page_numbers:
            pi = _as_int(p)
            if pi is not None:
                normalized_pages.append(pi)

        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else None
        except Exception:
            confidence = None

        item_warnings = item.get("warnings") if isinstance(item.get("warnings"), list) else []
        answers.append(
            {
                "question_id": qnum,
                "part_key": part,
                "answer_text": answer_text,
                "page_numbers": normalized_pages,
                "confidence": confidence,
                "needs_review": bool(item.get("needs_review", confidence is not None and confidence < 0.7)),
                "warnings": [str(w) for w in item_warnings if str(w).strip()],
            }
        )

    answers.sort(key=lambda a: (int(a.get("question_id") or 0), str(a.get("part_key") or "")))

    if not answers:
        warnings.append("No structured student answers were extracted.")

    extra_warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    warnings.extend(str(w) for w in extra_warnings if str(w).strip())

    return {
        "schema_version": "student_answer_bundle_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exam_id": str(data.get("exam_id") or ""),
        "student_name": str(data.get("student_name") or data.get("student") or ""),
        "student_id": str(data.get("student_id") or ""),
        "source_name": source_name,
        "document_type": document_type,
        "ocr_provider": provider,
        "ocr_model": model,
        "answers": answers,
        "unmatched_blocks": data.get("unmatched_blocks") if isinstance(data.get("unmatched_blocks"), list) else [],
        "warnings": warnings,
        "needs_teacher_review": bool(warnings) or any(a.get("needs_review") for a in answers),
    }


def parse_student_answer_bundle_from_model_text(
    text: str,
    *,
    source_name: str = "",
    provider: str = "",
    model: str | None = None,
    document_type: str = "unknown",
) -> dict[str, Any] | None:
    data = _extract_json_object(text)
    if not data:
        return None
    if "answers" not in data:
        return None
    return normalize_student_answer_bundle(
        data,
        source_name=source_name,
        provider=provider,
        model=model,
        document_type=document_type,
    )


def build_student_answer_bundle_from_tex(
    tex_path: Path,
    *,
    out_dir: Path,
    source_name: str = "",
    document_type: str = "tex",
) -> dict[str, Any]:
    answers_raw, _ranges = parse_student_tex_answers(tex_path, out_dir)
    data = {
        "source_name": source_name or tex_path.name,
        "document_type": document_type,
        "answers": [
            {
                "question_id": int(q),
                "part_key": part or "",
                "answer_text": body,
                "confidence": None,
                "warnings": [],
            }
            for (q, part), body in answers_raw.items()
            if str(body or "").strip()
        ],
    }
    return normalize_student_answer_bundle(data, source_name=source_name or tex_path.name, document_type=document_type)


def read_student_answers_from_bundle(bundle_path: Path) -> dict[tuple[int, str], str]:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle = normalize_student_answer_bundle(data, source_name=bundle_path.name)
    out: dict[tuple[int, str], str] = {}
    for a in bundle.get("answers") or []:
        qnum = _as_int(a.get("question_id"))
        if qnum is None:
            continue
        text = _clean_answer_text(a.get("answer_text"))
        if text:
            out[(qnum, _clean_part(a.get("part_key")))] = text
    return out


def student_answer_bundle_to_tex(bundle: dict[str, Any], *, source_name: str = "") -> str:
    bundle = normalize_student_answer_bundle(bundle, source_name=source_name or str(bundle.get("source_name") or ""))
    lines: list[str] = [
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
        r"% Student answer draft generated from student_answer_bundle.json.",
        rf"% Source: {source_name or bundle.get('source_name') or 'OCR upload'}",
        r"% This TeX is for review/display. Grading can use the JSON bundle directly.",
        "",
    ]

    current_q: int | None = None
    for a in bundle.get("answers") or []:
        qnum = _as_int(a.get("question_id"))
        if qnum is None:
            continue
        if current_q != qnum:
            lines.append("")
            lines.append(rf"\subsection*{{Question {qnum}}}")
            current_q = qnum
        part = _clean_part(a.get("part_key"))
        if part:
            lines.append(rf"\textbf{{({part})}}")
        text = _clean_answer_text(a.get("answer_text"))
        lines.append(text or "% Empty answer text")
        lines.append("")

    if not (bundle.get("answers") or []):
        lines.extend([
            r"% WARNING: No structured answers were found.",
            r"\subsection*{Question 1}",
            "",
        ])

    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def write_student_answer_bundle(bundle: dict[str, Any], out_dir: Path, *, filename: str = "student_answer_bundle.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
