from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from common.tex.part_normalize import normalize_part
from common.tex.student_tex import parse_student_tex_answers

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_model_json_text(text: str) -> str:
    cleaned = _CONTROL_CHARS_RE.sub(
        "",
        str(text or "").strip(),
    )
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()
    return cleaned


def _scan_complete_answer_objects(
    raw_text: str,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Recover complete answer objects from a truncated answer-bundle response.

    Gemini thinking and answer tokens share the same output budget. A long OCR
    response can therefore end in the middle of the last answer. We retain only
    fully closed answer objects and never invent the unfinished remainder.
    """
    text = _clean_model_json_text(raw_text)

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

        candidate = scan[object_start:index + 1]
        object_start = -1

        if '"answer_text"' not in candidate:
            continue

        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if (
            isinstance(parsed, dict)
            and str(parsed.get("answer_text") or "").strip()
        ):
            answers.append(parsed)

    return answers, depth > 0 or in_string


def _extract_answer_bundle_json(
    text: str,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Parse a normal answer bundle, or salvage complete answer objects when the
    provider response was truncated.
    """
    cleaned = _clean_model_json_text(text)

    if not cleaned or '"answers"' not in cleaned:
        return None, False

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])

            if (
                isinstance(data, dict)
                and isinstance(data.get("answers"), list)
            ):
                return data, False
        except Exception:
            pass

    recovered_answers, source_was_partial = (
        _scan_complete_answer_objects(cleaned)
    )

    if not recovered_answers:
        return None, source_was_partial

    return {
        "schema_version": "student_answer_bundle_v1",
        "answers": recovered_answers,
    }, True


def _flatten_embedded_answer_items(
    data: dict[str, Any],
    *,
    depth: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Flatten damaged wrappers such as one outer answer whose answer_text contains
    the real JSON answer bundle.
    """
    if depth > 5:
        return [], True

    raw_answers = data.get("answers")

    if not isinstance(raw_answers, list):
        return [], False

    flattened: list[dict[str, Any]] = []
    source_was_partial = False

    for answer in raw_answers:
        if not isinstance(answer, dict):
            continue

        answer_text = str(
            answer.get("answer_text")
            or answer.get("student_answer")
            or answer.get("text")
            or answer.get("raw_text")
            or ""
        ).strip()

        nested_bundle = None
        nested_partial = False

        if '"answers"' in answer_text:
            nested_bundle, nested_partial = (
                _extract_answer_bundle_json(answer_text)
            )

        source_was_partial = (
            source_was_partial
            or nested_partial
        )

        if nested_bundle:
            nested_answers, deeper_partial = (
                _flatten_embedded_answer_items(
                    nested_bundle,
                    depth=depth + 1,
                )
            )
            source_was_partial = (
                source_was_partial
                or deeper_partial
            )

            if nested_answers:
                flattened.extend(nested_answers)
                continue

        flattened.append(dict(answer))

    return flattened, source_was_partial


def _is_comment_only_answer(value: Any) -> bool:
    text = str(value or "").strip()

    if not text:
        return True

    nonempty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return bool(nonempty_lines) and all(
        line == "%" or line.startswith("%")
        for line in nonempty_lines
    )


def recover_student_answer_bundle_from_text(
    text: str,
    *,
    source_name: str = "",
    provider: str = "",
    model: str | None = None,
    document_type: str = "unknown",
) -> tuple[dict[str, Any] | None, bool]:
    """
    Provider-neutral structured OCR recovery used by both OCR ingestion and
    grading. Returns the normalized bundle and whether the source was partial.
    """
    data, source_was_partial = _extract_answer_bundle_json(text)

    if not data:
        return None, source_was_partial

    if source_was_partial:
        data = dict(data)
        data["warnings"] = list(data.get("warnings") or []) + [
            (
                "The OCR provider response ended before the full JSON bundle "
                "was complete. Fully closed answer blocks were recovered; "
                "later unfinished work may be missing."
            )
        ]

    bundle = normalize_student_answer_bundle(
        data,
        source_name=source_name,
        provider=provider,
        model=model,
        document_type=document_type,
    )

    return bundle, source_was_partial
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


def _clean_string_list(
    value: Any,
    *,
    max_items: int = 20,
    max_length: int = 800,
) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    cleaned: list[str] = []

    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in cleaned:
            continue

        cleaned.append(text[:max_length])

        if len(cleaned) >= max_items:
            break

    return cleaned


def _normalize_visual_capture_status(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")

    if normalized in {"complete", "partial", "none", "uncertain"}:
        return normalized

    return "uncertain"


def _clamp_unit(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None

    if number < 0 or number > 1:
        return None

    return round(number, 6)


def _normalize_region(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    page_number = _as_int(
        value.get("page_number")
        or value.get("page")
        or value.get("page_index")
    )

    # Accept either x/y/width/height or a four-value bbox.
    bbox = value.get("bbox") or value.get("normalized_bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x_raw, y_raw, width_raw, height_raw = bbox[:4]
    else:
        x_raw = value.get("x", value.get("left"))
        y_raw = value.get("y", value.get("top"))
        width_raw = value.get("width")
        height_raw = value.get("height")

        if width_raw is None and value.get("right") is not None:
            try:
                width_raw = float(value.get("right")) - float(x_raw)
            except Exception:
                width_raw = None

        if height_raw is None and value.get("bottom") is not None:
            try:
                height_raw = float(value.get("bottom")) - float(y_raw)
            except Exception:
                height_raw = None

    x = _clamp_unit(x_raw)
    y = _clamp_unit(y_raw)
    width = _clamp_unit(width_raw)
    height = _clamp_unit(height_raw)

    if page_number is None or page_number < 1:
        return None

    if None in (x, y, width, height):
        return None

    if width <= 0 or height <= 0:
        return None

    # Keep the box inside the page even when a model rounds past the edge.
    width = round(min(width, 1 - x), 6)
    height = round(min(height, 1 - y), 6)

    if width <= 0 or height <= 0:
        return None

    confidence_raw = value.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except Exception:
        confidence = None

    return {
        "page_number": page_number,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "text_excerpt": str(
            value.get("text_excerpt")
            or value.get("excerpt")
            or value.get("text")
            or ""
        ).strip()[:500],
        "confidence": confidence,
    }


def _normalize_regions(value: Any) -> list[dict[str, Any]]:
    raw_regions = value if isinstance(value, list) else []
    regions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for raw_region in raw_regions:
        region = _normalize_region(raw_region)
        if not region:
            continue

        signature = (
            region["page_number"],
            region["x"],
            region["y"],
            region["width"],
            region["height"],
        )

        if signature in seen:
            continue

        seen.add(signature)
        regions.append(region)

    regions.sort(
        key=lambda item: (
            int(item.get("page_number") or 1),
            float(item.get("y") or 0),
            float(item.get("x") or 0),
        )
    )
    return regions


def normalize_student_answer_bundle(
    data: dict[str, Any] | None,
    *,
    source_name: str = "",
    provider: str = "",
    model: str | None = None,
    document_type: str = "unknown",
) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    answers: list[dict[str, Any]] = []
    warnings: list[str] = []

    raw_answers, embedded_source_was_partial = (
        _flatten_embedded_answer_items(data)
    )

    if embedded_source_was_partial:
        warnings.append(
            (
                "Recovered complete answer blocks from an embedded or "
                "truncated OCR JSON bundle."
            )
        )

    if not isinstance(data.get("answers"), list):
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
        visual_elements = _clean_string_list(
            item.get("visual_elements")
            or item.get("visual_descriptions")
            or []
        )
        visual_capture_status = _normalize_visual_capture_status(
            item.get("visual_capture_status")
        )

        if _is_comment_only_answer(answer_text):
            warnings.append(
                f"Skipped Q{qnum}{part or ''}: parser comment/debris."
            )
            continue

        if not answer_text and not visual_elements:
            warnings.append(f"Skipped Q{qnum}{part or ''}: empty answer_text.")
            continue

        regions = _normalize_regions(
            item.get("regions")
            or item.get("answer_regions")
            or []
        )

        page_numbers = item.get("page_numbers") or item.get("pages") or []
        if not isinstance(page_numbers, list):
            page_numbers = []

        normalized_pages: list[int] = []
        for p in page_numbers:
            pi = _as_int(p)
            if pi is not None and pi > 0 and pi not in normalized_pages:
                normalized_pages.append(pi)

        for region in regions:
            page_number = int(region["page_number"])
            if page_number not in normalized_pages:
                normalized_pages.append(page_number)

        normalized_pages.sort()

        key = (qnum, part)
        if key in seen:
            # Merge duplicate fragments conservatively, including their page
            # regions so split answer blocks remain usable by the overlay.
            for existing in answers:
                if existing["question_id"] == qnum and existing.get("part_key", "") == part:
                    existing["answer_text"] = (
                        existing["answer_text"].rstrip()
                        + "\n\n"
                        + answer_text
                    ).strip()

                    merged_pages = sorted(
                        set(existing.get("page_numbers") or [])
                        | set(normalized_pages)
                    )
                    existing["page_numbers"] = merged_pages

                    existing["regions"] = _normalize_regions(
                        list(existing.get("regions") or [])
                        + regions
                    )

                    merged_visual_elements = _clean_string_list(
                        list(existing.get("visual_elements") or [])
                        + visual_elements
                    )
                    existing["visual_elements"] = merged_visual_elements

                    existing_status = _normalize_visual_capture_status(
                        existing.get("visual_capture_status")
                    )

                    if "uncertain" in {
                        existing_status,
                        visual_capture_status,
                    }:
                        existing["visual_capture_status"] = "uncertain"
                    elif "partial" in {
                        existing_status,
                        visual_capture_status,
                    }:
                        existing["visual_capture_status"] = "partial"
                    elif merged_visual_elements:
                        existing["visual_capture_status"] = "complete"
                    else:
                        existing["visual_capture_status"] = "none"

                    existing.setdefault("warnings", []).append(
                        "Merged duplicate OCR block for this question/part."
                    )
                    break
            continue
        seen.add(key)

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
                "visual_elements": visual_elements,
                "visual_capture_status": visual_capture_status,
                "page_numbers": normalized_pages,
                "regions": regions,
                "confidence": confidence,
                "needs_review": bool(
                    item.get(
                        "needs_review",
                        (
                            confidence is not None
                            and confidence < 0.7
                        )
                        or visual_capture_status in {"partial", "uncertain"},
                    )
                ),
                "warnings": [
                    str(w)
                    for w in item_warnings
                    if str(w).strip()
                ],
            }
        )

    answers.sort(key=lambda a: (int(a.get("question_id") or 0), str(a.get("part_key") or "")))

    if not answers:
        warnings.append("No structured student answers were extracted.")

    extra_warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    warnings.extend(str(w) for w in extra_warnings if str(w).strip())

    resolved_source_name = (
        str(source_name or "").strip()
        or str(data.get("source_name") or "").strip()
    )
    resolved_document_type = (
        str(document_type or "").strip()
        if str(document_type or "").strip() not in {"", "unknown"}
        else str(data.get("document_type") or "unknown").strip()
    )
    resolved_provider = (
        str(provider or "").strip()
        or str(data.get("ocr_provider") or "").strip()
    )
    resolved_model = (
        model
        if model not in (None, "")
        else data.get("ocr_model")
    )

    return {
        "schema_version": "student_answer_bundle_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exam_id": str(data.get("exam_id") or ""),
        "student_name": str(data.get("student_name") or data.get("student") or ""),
        "student_id": str(data.get("student_id") or ""),
        "source_name": resolved_source_name,
        "document_type": resolved_document_type,
        "ocr_provider": resolved_provider,
        "ocr_model": resolved_model,
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
    bundle, _source_was_partial = (
        recover_student_answer_bundle_from_text(
            text,
            source_name=source_name,
            provider=provider,
            model=model,
            document_type=document_type,
        )
    )
    return bundle

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


def read_student_answers_from_bundle(
    bundle_path: Path,
) -> dict[tuple[int, str], str]:
    records = read_student_answer_records_from_bundle(bundle_path)

    return {
        key: str(record.get("answer_text") or "").strip()
        for key, record in records.items()
        if str(record.get("answer_text") or "").strip()
    }


def read_student_answer_records_from_bundle(
    bundle_path: Path,
) -> dict[tuple[int, str], dict[str, Any]]:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle = normalize_student_answer_bundle(
        data,
        source_name=bundle_path.name,
    )
    out: dict[tuple[int, str], dict[str, Any]] = {}

    for answer in bundle.get("answers") or []:
        qnum = _as_int(answer.get("question_id"))

        if qnum is None:
            continue

        key = (
            qnum,
            _clean_part(answer.get("part_key")),
        )

        out[key] = {
            "answer_text": _clean_answer_text(
                answer.get("answer_text")
            ),
            "visual_elements": _clean_string_list(
                answer.get("visual_elements") or []
            ),
            "visual_capture_status": _normalize_visual_capture_status(
                answer.get("visual_capture_status")
            ),
            "ocr_confidence": answer.get("confidence"),
            "ocr_needs_review": bool(
                answer.get("needs_review")
            ),
            "ocr_warnings": _clean_string_list(
                answer.get("warnings") or [],
                max_items=10,
                max_length=500,
            ),
            "page_numbers": list(
                answer.get("page_numbers") or []
            ),
            "regions": list(
                answer.get("regions") or []
            ),
        }

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