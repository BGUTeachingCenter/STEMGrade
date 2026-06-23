# core/ai_clients/model_capabilities.py
"""
Central, model-aware capability hints.

Goal: one OcrOptions (with a single `temperature`) can drive many different
models/providers without hand-tweaking parameters per call.

Two layers of protection:

1. Static defaults here decide whether to even *send* a parameter. For example,
   OpenAI reasoning models (o1/o3/o4) and the gpt-5 family reject a custom
   sampling temperature and only run at their default — so we omit it.
2. The clients additionally retry the request without the offending parameter if
   the provider rejects it at runtime (see gpt_client._post_responses). That way
   an unknown/new model still works even if it is not listed here.

Per-call overrides always win, via OcrOptions.extra:

    options.extra["send_temperature"] = False   # force OFF
    options.extra["send_temperature"] = True    # force ON

No-code escape hatch for new models that only accept the default temperature:

    OPENAI_FIXED_TEMPERATURE_MODELS="gpt-5.5,my-new-reasoner"
"""
from __future__ import annotations

import os
import re
from typing import Optional

from schemas.ocr_response import OcrOptions


# OpenAI model-name prefixes that only support the default temperature.
_OPENAI_FIXED_TEMPERATURE = re.compile(r"^(o1|o3|o4|gpt-5)", re.IGNORECASE)


def normalize_provider(provider: str) -> str:
    """Collapse the many provider aliases used around the codebase."""
    p = (provider or "").strip().lower()
    if p in {"openai", "gpt", "chatgpt", "oai"}:
        return "openai"
    if p in {"google", "gemini", "ai_studio", "aistudio"}:
        return "google"
    if p in {"ollama", "local"}:
        return "ollama"
    if p == "mathpix":
        return "mathpix"
    return p


def _openai_has_fixed_temperature(model: str) -> bool:
    model = (model or "").strip()
    if _OPENAI_FIXED_TEMPERATURE.match(model):
        return True

    # Env extension: comma-separated prefixes, matched case-insensitively.
    extra = os.getenv("OPENAI_FIXED_TEMPERATURE_MODELS", "")
    prefixes = [p.strip().lower() for p in extra.split(",") if p.strip()]
    ml = model.lower()
    return any(ml.startswith(p) for p in prefixes)


def supports_temperature(provider: str, model: str) -> bool:
    """Best-effort default: does this provider/model accept a *custom* temperature?"""
    provider = normalize_provider(provider)

    if provider == "openai":
        return not _openai_has_fixed_temperature(model)

    # Gemini / Ollama / others accept a temperature. Mathpix has no LLM
    # sampling, but it never reads this value anyway.
    return True


def resolve_temperature(
    *,
    provider: str,
    model: str,
    options: OcrOptions,
) -> Optional[float]:
    """
    Decide the temperature to actually send, or None to omit it entirely.

    Resolution order:
      1. Explicit per-call override: options.extra["send_temperature"] (bool)
      2. options.temperature is None -> omit (use the model's own default)
      3. Static capability default for this provider/model
    """
    override = options.extra.get("send_temperature") if options.extra else None
    if override is False:
        return None

    if options.temperature is None:
        return None

    if override is True:
        return float(options.temperature)

    if supports_temperature(provider, model):
        return float(options.temperature)

    return None
