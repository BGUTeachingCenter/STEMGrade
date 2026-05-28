from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

MATHPIX_TEXT_URL = "https://api.mathpix.com/v3/text"
MATHPIX_PDF_URL = "https://api.mathpix.com/v3/pdf"


class MathpixError(RuntimeError):
    pass


def _headers(app_id: str, app_key: str) -> dict[str, str]:
    return {
        "app_id": app_id,
        "app_key": app_key,
    }


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_guess_mime(path)};base64,{b64}"


def _json_or_error(resp: requests.Response, context: str) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception as e:
        raise MathpixError(f"{context}: Mathpix returned non-JSON response: {e}") from e

    if resp.status_code >= 400:
        err = data.get("error") or data.get("error_info") or data
        raise MathpixError(f"{context}: Mathpix failed: {err}")

    if data.get("error"):
        raise MathpixError(f"{context}: Mathpix error: {data['error']}")

    return data


def _process_image(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    include_line_data: bool = True,
) -> dict[str, Any]:
    headers = {
        **_headers(app_id, app_key),
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "src": _to_data_url(file_path),
        "math_inline_delimiters": ["$", "$"],
        "rm_spaces": True,
        "include_latex": True,
        "include_line_data": include_line_data,
        "formats": ["text", "data", "html", "latex_styled"],
        "data_options": {"include_latex": True},
        "metadata": {"improve_mathpix": False},
    }

    try:
        resp = requests.post(MATHPIX_TEXT_URL, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise MathpixError(f"Could not reach Mathpix image OCR API: {e}") from e

    data = _json_or_error(resp, "Image OCR")
    data["_mathpix_mode"] = "image"
    return data


def _submit_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
) -> str:
    options = {
        "rm_spaces": True,
        "rm_fonts": True,
        "conversion_formats": {
            "tex.zip": True,
        },
        "metadata": {
            "improve_mathpix": False,
        },
    }

    try:
        with file_path.open("rb") as f:
            resp = requests.post(
                MATHPIX_PDF_URL,
                headers=_headers(app_id, app_key),
                files={"file": (file_path.name, f, "application/pdf")},
                data={"options_json": json.dumps(options)},
                timeout=120,
            )
    except requests.RequestException as e:
        raise MathpixError(f"Could not submit PDF to Mathpix: {e}") from e

    data = _json_or_error(resp, "PDF submit")
    pdf_id = data.get("pdf_id")
    if not pdf_id:
        raise MathpixError(f"PDF submit did not return pdf_id: {data}")

    return str(pdf_id)


def _poll_pdf_until_done(
    *,
    pdf_id: str,
    app_id: str,
    app_key: str,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{MATHPIX_PDF_URL}/{pdf_id}",
                headers=_headers(app_id, app_key),
                timeout=60,
            )
        except requests.RequestException as e:
            raise MathpixError(f"Could not poll Mathpix PDF status: {e}") from e

        data = _json_or_error(resp, "PDF status")
        last_status = data
        status = str(data.get("status") or "").lower()

        if status == "completed":
            return data

        if status in {"error", "failed"}:
            raise MathpixError(f"Mathpix PDF OCR failed: {data}")

        time.sleep(poll_interval_seconds)

    raise MathpixError(f"Mathpix PDF OCR timed out. Last status: {last_status}")


def _download_pdf_text(
    *,
    pdf_id: str,
    app_id: str,
    app_key: str,
) -> tuple[str, str]:
    """
    Prefer MMD because Mathpix PDF OCR naturally returns Mathpix Markdown.
    Fallback to md if the account/API returns that extension instead.
    """
    last_error = ""

    for ext in ("mmd", "md"):
        try:
            resp = requests.get(
                f"{MATHPIX_PDF_URL}/{pdf_id}.{ext}",
                headers=_headers(app_id, app_key),
                timeout=120,
            )
        except requests.RequestException as e:
            last_error = str(e)
            continue

        if resp.status_code == 200 and resp.text.strip():
            return resp.text, ext

        last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"

    raise MathpixError(f"Could not download Mathpix PDF text output. Last error: {last_error}")


def _process_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
) -> dict[str, Any]:
    pdf_id = _submit_pdf(file_path=file_path, app_id=app_id, app_key=app_key)
    status = _poll_pdf_until_done(pdf_id=pdf_id, app_id=app_id, app_key=app_key)
    text, ext = _download_pdf_text(pdf_id=pdf_id, app_id=app_id, app_key=app_key)

    return {
        "_mathpix_mode": "pdf",
        "pdf_id": pdf_id,
        "status": status,
        "downloaded_ext": ext,
        "text": text,
        "html": "",
        "latex_styled": "",
        "line_data": [],
        "confidence": None,
        "is_handwritten": True,
    }


_Q_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:Question|Q|שאלה|תרגיל)\s*([0-9]{1,2})(?:\s*[\).:\-–]\s*([A-Za-zא-ת]))?.*$",
    re.IGNORECASE | re.UNICODE,
)

_Q_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?([0-9]{1,2})\s*[\).:]\s*$",
    re.UNICODE,
)

_PART_RE = re.compile(
    r"^\s*(?:סעיף\s*)?[\(\[]?\s*([A-Za-zא-ת])\s*[\)\].:]\s*(.*)$",
    re.UNICODE,
)


def _normalize_ocr_lines_for_student_parser(text: str) -> str:
    """
    Convert OCR text/MMD into the markers already supported by:
    grader.file_handling.student_tex.parse_student_tex_answers

    Important parser markers:
      \\subsection*{Question 1}
      \\textbf{(א)}
      \\textbf{(a)}
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    out: list[str] = []
    seen_q: set[int] = set()
    detected_any_question = False

    for raw in lines:
        line = raw.strip()

        if not line:
            out.append("")
            continue

        q_match = _Q_RE.match(line)
        if q_match:
            qnum = int(q_match.group(1))
            part = q_match.group(2)

            if qnum not in seen_q:
                out.append("")
                out.append(rf"\subsection*{{Question {qnum}}}")
                seen_q.add(qnum)
                detected_any_question = True

            if part:
                out.append(rf"\textbf{{({part})}}")
            continue

        q_line_match = _Q_LINE_RE.match(line)
        if q_line_match:
            qnum = int(q_line_match.group(1))

            if 1 <= qnum <= 99 and qnum not in seen_q:
                out.append("")
                out.append(rf"\subsection*{{Question {qnum}}}")
                seen_q.add(qnum)
                detected_any_question = True
                continue

        part_match = _PART_RE.match(line)
        if part_match:
            part = part_match.group(1)
            rest = part_match.group(2).strip()
            out.append(rf"\textbf{{({part})}}")
            if rest:
                out.append(rest)
            continue

        out.append(line)

    body = "\n".join(out).strip()

    if not detected_any_question:
        body = "\n".join(
            [
                r"% WARNING: MathGrade could not confidently detect question titles.",
                r"% Before grading, split this OCR text into:",
                r"% \subsection*{Question 1}",
                r"% \textbf{(א)}",
                r"% \textbf{(ב)}",
                "",
                r"\subsection*{Question 1}",
                "",
                body,
            ]
        ).strip()

    return body


def mathpix_text_to_student_tex(text: str, *, source_name: str = "") -> str:
    body = _normalize_ocr_lines_for_student_parser(text)

    return "\n".join(
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
            r"% OCR draft generated by MathGrade + Mathpix.",
            rf"% Source: {source_name}" if source_name else r"% Source: OCR upload",
            r"% Review carefully before grading.",
            r"% Required grading markers:",
            r"%   \subsection*{Question 1}",
            r"%   \textbf{(א)}",
            r"%   \textbf{(ב)}",
            "",
            body,
            "",
            r"\end{document}",
            "",
        ]
    )


def process_image_or_pdf(
    *,
    file_path: Path,
    app_id: str,
    app_key: str,
    include_line_data: bool = True,
) -> dict[str, Any]:
    if not app_id or not app_key:
        raise MathpixError("Missing Mathpix credentials.")

    if not file_path.exists():
        raise MathpixError(f"OCR input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _process_pdf(file_path=file_path, app_id=app_id, app_key=app_key)

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return _process_image(
            file_path=file_path,
            app_id=app_id,
            app_key=app_key,
            include_line_data=include_line_data,
        )

    raise MathpixError(f"Unsupported OCR file type: {suffix}")