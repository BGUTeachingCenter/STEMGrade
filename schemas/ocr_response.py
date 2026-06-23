from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


OcrProvider = Literal["mathpix", "openai", "google", "tex", "unknown"]
OcrInputKind = Literal["pdf", "image", "tex", "text", "unknown"]


class AiUsage(BaseModel):
    """
    Provider-neutral usage information.

    OpenAI:
      input_tokens / output_tokens / total_tokens

    Google/Gemini:
      promptTokenCount -> input_tokens
      candidatesTokenCount -> output_tokens
      totalTokenCount -> total_tokens

    Mathpix:
      usually zero, because Mathpix does not report LLM-style token usage.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    raw_usage: dict[str, Any] = Field(default_factory=dict)


class OcrOptions(BaseModel):
    """
    Provider-neutral OCR options.

    Each provider should use the options it supports and ignore the rest.
    This allows easy model/provider switching without changing route code.
    """

    # Sampling temperature.
    #   None  -> never send a temperature (use the model's own default).
    #   float -> the clients send it ONLY when the target model accepts a custom
    #            temperature (see core.ai_clients.model_capabilities). gpt-5 /
    #            o-series reject it, so it is omitted automatically.
    # Force on/off per call with extra={"send_temperature": True/False}.
    temperature: Optional[float] = 0.0
    max_output_tokens: int = 12000
    timeout_s: int = 300

    language_hint: str = "hebrew"
    preserve_math: bool = True
    preserve_layout: bool = True
    include_line_data: bool = True

    prompt_variant: str = "default"

    # Escape hatch for provider-specific experiments.
    extra: dict[str, Any] = Field(default_factory=dict)


class OcrPage(BaseModel):
    page_index: int = 0
    page_number: int = 1

    text: str = ""
    html: str = ""
    latex_styled: str = ""

    confidence: Optional[float] = None
    line_data: list[dict[str, Any]] = Field(default_factory=list)


class OcrResponse(BaseModel):
    """
    Universal OCR response.

    This is intentionally naive. It does not know whether the uploaded file is:
      - teacher exam/questions
      - teacher official solution
      - student handwritten work
      - future image-based question

    Task-specific interpretation happens downstream.
    """

    schema_version: str = "ocr_response_v1"
    ok: bool = True

    provider: OcrProvider = "unknown"
    model: Optional[str] = None

    input_kind: OcrInputKind = "unknown"
    source_filename: str = ""
    source_path: str = ""

    text: str = ""
    pages: list[OcrPage] = Field(default_factory=list)

    html: str = ""
    latex_styled: str = ""

    confidence: Optional[float] = None
    is_handwritten: Optional[bool] = None

    provider_mode: Optional[str] = None
    provider_document_id: Optional[str] = None
    provider_status: Optional[Any] = None

    response_id: Optional[str] = None

    usage: AiUsage = Field(default_factory=AiUsage)

    raw: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def primary_text(self) -> str:
        if self.text:
            return self.text
        return "\n\n".join(p.text for p in self.pages if p.text).strip()

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Compatibility adapter for old code that expects Mathpix/OpenAI-like dicts.
        This lets us migrate gradually.
        """
        text = self.primary_text()

        return {
            "text": text,
            "html": self.html,
            "latex_styled": self.latex_styled,
            "line_data": self.pages[0].line_data if self.pages else [],
            "confidence": self.confidence,
            "is_handwritten": self.is_handwritten,

            # Old Mathpix-like keys
            "_mathpix_mode": self.provider_mode if self.provider == "mathpix" else None,
            "pdf_id": self.provider_document_id if self.provider == "mathpix" else None,
            "status": self.provider_status,
            "downloaded_ext": (self.raw or {}).get("downloaded_ext") or "txt",

            # Old OpenAI-like keys
            "_openai_mode": self.provider_mode if self.provider == "openai" else None,
            "model": self.model,
            "response_id": self.response_id,

            # New neutral keys
            "schema_version": self.schema_version,
            "provider": self.provider,
            "input_kind": self.input_kind,
            "source_filename": self.source_filename,
            "source_path": self.source_path,
            "provider_mode": self.provider_mode,
            "provider_document_id": self.provider_document_id,
            "provider_status": self.provider_status,
            "warnings": self.warnings,
            "raw": self.raw,
            "usage": self.usage.model_dump(),
        }


def guess_input_kind(path: Path) -> OcrInputKind:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return "pdf"

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"

    if suffix == ".tex":
        return "tex"

    if suffix == ".txt":
        return "text"

    return "unknown"