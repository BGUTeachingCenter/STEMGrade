from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from schemas.ocr_response import OcrOptions, OcrResponse


class OcrClientError(RuntimeError):
    pass


def run_ocr(
    *,
    file_path: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    options: Optional[OcrOptions] = None,
) -> OcrResponse:
    """
    Universal OCR entrypoint.

    All routes/tasks should eventually call only this function.

    Provider examples:
      provider="mathpix"
      provider="openai"
      provider="google"

    The returned object is always OcrResponse.
    """

    provider = (provider or os.getenv("OCR_PROVIDER") or "mathpix").strip().lower()
    options = options or OcrOptions()

    if provider == "mathpix":
        from core.ai_clients.ocr_mathpix_client import run_mathpix_ocr

        return run_mathpix_ocr(
            file_path=file_path,
            model=model,
            options=options,
        )

    if provider in {"openai", "gpt", "chatgpt"}:
        from core.ai_clients.gpt_client import GptClient

        client = GptClient(model=model or None)
        return client.ocr_document(
            file_path=file_path,
            model=model or None,
            options=options,
        )

    if provider in {"google", "gemini", "ai_studio", "aistudio"}:
        from core.ai_clients.google_client import GoogleClient

        client = GoogleClient(model=model or None)

        return client.ocr_document(
            file_path=file_path,
            model=model or None,
            options=options,
        )

    raise OcrClientError(f"Unsupported OCR provider: {provider}")