# services/ai_grading/google_client.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from core.ai_clients.ai_usage_logger import log_ai_usage
from core.ai_clients.model_capabilities import resolve_temperature
from core.ai_clients.shared import (
    build_ocr_prompt,
    ensure_supported_ocr_file,
    file_to_base64,
    guess_mime,
    safe_json_loads,
)
from schemas.ocr_response import AiUsage, OcrOptions, OcrPage, OcrResponse, guess_input_kind

import requests






def _resolve_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline all local ``$ref`` pointers so Gemini never sees ``$ref``/``$defs``.

    Pydantic v2 ``model_json_schema()`` emits nested models as ``$ref`` pointers
    into a ``$defs`` (or ``definitions``) section. Gemini's responseSchema does
    not support either, so we replace every ``#/$defs/Name`` reference with a
    copy of the referenced definition (recursively). Self-referential models are
    guarded against by tracking the refs currently being expanded.
    """
    defs: Dict[str, Any] = {}
    if isinstance(schema, dict):
        defs = {**(schema.get("definitions") or {}), **(schema.get("$defs") or {})}

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.split("/")[-1]
                if name in defs and name not in seen:
                    resolved = resolve(defs[name], seen | {name})
                    # Merge any sibling keys of $ref (e.g. description) onto the
                    # resolved definition, without letting $ref itself survive.
                    extra = {k: resolve(v, seen) for k, v in node.items() if k != "$ref"}
                    if isinstance(resolved, dict):
                        return {**resolved, **extra}
                    return resolved
                # Unknown/circular ref: drop it, leaving a permissive object.
                return {"type": "object"}
            return {k: resolve(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(i, seen) for i in node]
        return node

    return resolve(schema, frozenset())


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Gemini responseSchema is a restricted subset. Remove keys it rejects.
    DROP_KEYS = {
        "additionalProperties",
        "$schema", "$id", "$ref", "definitions", "$defs",
        "patternProperties", "propertyNames",
        "dependencies", "dependentSchemas", "dependentRequired",
        "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
        "examples", "default", "format",
        "readOnly", "writeOnly", "nullable",
    }

    def _collapse_union(members: Any) -> Dict[str, Any] | None:
        # Pydantic renders `X | None` as anyOf/oneOf with a {"type": "null"}
        # branch. Gemini has no such union, so collapse to the first concrete
        # (non-null) branch instead of dropping the whole property.
        if not isinstance(members, list):
            return None
        for member in members:
            if isinstance(member, dict) and member.get("type") != "null":
                return member
        return None

    # Inline $ref/$defs before stripping so we don't leave dangling references.
    resolved = _resolve_refs(schema or {})

    def rec(x: Any) -> Any:
        if isinstance(x, dict):
            union = _collapse_union(x.get("anyOf") or x.get("oneOf"))
            if union is not None:
                # Merge sibling keys (title, description, ...) onto the chosen
                # branch, then recurse so it gets cleaned like any other node.
                siblings = {
                    k: v for k, v in x.items()
                    if k not in {"anyOf", "oneOf"} and k not in DROP_KEYS
                }
                return rec({**siblings, **union})
            return {k: rec(v) for k, v in x.items() if k not in DROP_KEYS}
        if isinstance(x, list):
            return [rec(i) for i in x]
        return x

    cleaned = rec(resolved)
    if isinstance(cleaned, dict) and "type" not in cleaned:
        cleaned["type"] = "object"
    return cleaned


def _chat_json_max_output_tokens() -> int:
    """Output-token budget for chat_json.

    Gemini's default output cap is small enough to truncate a full structured
    bundle (Hebrew text + LaTeX is token-heavy). Default high; override with
    GEMINI_CHAT_MAX_OUTPUT_TOKENS / GOOGLE_CHAT_MAX_OUTPUT_TOKENS.
    """
    raw = (
        os.getenv("GEMINI_CHAT_MAX_OUTPUT_TOKENS")
        or os.getenv("GOOGLE_CHAT_MAX_OUTPUT_TOKENS")
        or ""
    ).strip()
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    # 65536 is the max output for the Gemini 2.5 family; a full solution bundle
    # (many questions, each echoing question text + solution as JSON) can need
    # far more than 32k output tokens.
    return 65536




def _file_to_inline_data(path: Path) -> dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": guess_mime(path),
            "data": file_to_base64(path),
        }
    }



def _extract_gemini_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []

    for cand in data.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)

    return "\n".join(chunks).strip()


def _extract_google_usage(data: dict[str, Any]) -> AiUsage:
    usage = data.get("usageMetadata") or {}

    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens) or 0)
    reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)

    return AiUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage,
    )

@dataclass
class GoogleClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model = (model or os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "").strip()
        self.api_base = (os.getenv("GOOGLE_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
        self.api_version = (os.getenv("GOOGLE_API_VERSION") or "v1beta").strip()

        # ✅ token counters (Gemini usageMetadata)
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_candidate_tokens = 0
        self.total_thoughts_tokens = 0
        self.last_usage: dict[str, Any] = {}


    def _record_usage(
        self,
        *,
        data: dict[str, Any],
        task: str,
        source_filename: str = "",
        input_kind: str = "",
        response_id: str = "",
    ) -> AiUsage:
        usage = _extract_google_usage(data)

        self.total_prompt_tokens += usage.input_tokens
        self.total_candidate_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_thoughts_tokens += usage.reasoning_tokens

        self.last_usage = usage.model_dump()

        log_ai_usage(
            {
                "task": task,
                "provider": "google",
                "model": self.model,
                "source_filename": source_filename,
                "input_kind": input_kind,
                "response_id": response_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "raw_usage": usage.raw_usage,
            }
        )

        return usage

    def chat_json(
            self,
            *,
            system: str,
            user: str,
            schema: Dict[str, Any],
            temperature: float = 0.15,
            timeout_s: int = 120,
            schema_name: str = "result",
            strict: bool = False,
    ) -> dict:
        """
        Return JSON from Gemini.

        schema_name and strict are accepted for interface compatibility with
        GptClient.chat_json(). Gemini does not use these exact OpenAI fields;
        the schema is sent as responseSchema after Gemini-specific sanitizing.
        """
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")
        if not self.model:
            raise RuntimeError("Missing GOOGLE_MODEL (or GEMINI_MODEL), or set client.model before calling.")

        url = f"{self.api_base}/{self.api_version}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        safe_schema = _sanitize_schema_for_gemini(schema)

        generation_config: dict[str, Any] = {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
            "responseSchema": safe_schema,
            # Without an explicit cap Gemini uses a small default output budget,
            # which truncates long JSON (e.g. a full Hebrew exam bundle) mid-token
            # and yields unparseable output. Give it plenty of room.
            "maxOutputTokens": _chat_json_max_output_tokens(),
        }

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        r = requests.post(url, params=params, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"Google Gemini API error {r.status_code}: {r.text[:1000]}")

        data = r.json()

        # ✅ accumulate usageMetadata if present
        self._record_usage(
            data=data,
            task="chat_json",
        )

        candidate = (data.get("candidates") or [{}])[0]
        finish_reason = str(candidate.get("finishReason") or "").upper()

        text = _extract_gemini_text(data)

        if finish_reason == "MAX_TOKENS":
            usage = data.get("usageMetadata") or {}
            thoughts = usage.get("thoughtsTokenCount")
            thoughts_note = (
                f", thoughtsTokenCount={thoughts} (thinking also consumes the budget)"
                if thoughts
                else ""
            )
            raise RuntimeError(
                "Gemini response was truncated (finishReason=MAX_TOKENS): the model "
                "hit its output-token limit before finishing the JSON. "
                f"Budget was {generation_config['maxOutputTokens']} output tokens; "
                f"promptTokenCount={usage.get('promptTokenCount')}, "
                f"candidatesTokenCount={usage.get('candidatesTokenCount')}{thoughts_note}. "
                "Raise GEMINI_CHAT_MAX_OUTPUT_TOKENS (max 65536 for Gemini 2.5), or "
                "split the input into fewer questions per request."
            )

        if not text:
            raise RuntimeError(
                f"Gemini returned empty text (finishReason={finish_reason or 'unknown'}). "
                "First 800 chars of response:\n"
                + json.dumps(data, ensure_ascii=False)[:800]
            )

        try:
            return safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Gemini returned non-JSON content (finishReason={finish_reason or 'unknown'}). "
                f"First 500 chars:\n{text[:500]}"
            ) from e


    def ocr_document(
        self,
        *,
        file_path: Path,
        model: Optional[str] = None,
        options: Optional[OcrOptions] = None,
    ) -> OcrResponse:
        """
        OCR-like extraction using Gemini / Google AI Studio.

        This belongs here, not in ocr_google_client.py, because it uses the same
        Google API key, model config, endpoint config, and token logging.
        """
        options = options or OcrOptions()

        model_to_use = (
            model
            or os.getenv("GOOGLE_OCR_MODEL")
            or os.getenv("GEMINI_OCR_MODEL")
            or self.model
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("GEMINI_MODEL")
            or ""
        ).strip()

        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")

        if not model_to_use:
            raise RuntimeError("Missing GOOGLE_OCR_MODEL / GEMINI_OCR_MODEL / GOOGLE_MODEL / GEMINI_MODEL.")

        suffix = ensure_supported_ocr_file(
            file_path,
            provider="Google",
        )

        url = f"{self.api_base}/{self.api_version}/models/{model_to_use}:generateContent"
        params = {"key": self.api_key}

        generation_config: dict[str, Any] = {
            "maxOutputTokens": int(options.max_output_tokens),
        }

        # Only send a temperature when this model accepts a custom one.
        temperature = resolve_temperature(provider="google", model=model_to_use, options=options)
        if temperature is not None:
            generation_config["temperature"] = temperature

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": build_ocr_prompt(options)},
                        _file_to_inline_data(file_path),
                    ],
                }
            ],
            "generationConfig": generation_config,
        }

        # Escape hatch for OCR experiments.
        for k, v in (options.extra.get("google_payload") or {}).items():
            payload[k] = v

        r = requests.post(
            url,
            params=params,
            json=payload,
            timeout=options.timeout_s,
        )

        if r.status_code != 200:
            raise RuntimeError(f"Google Gemini OCR error {r.status_code}: {r.text[:2000]}")

        data = r.json()

        candidate = (
            (data.get("candidates") or [{}])[0]
        )
        finish_reason = str(
            candidate.get("finishReason") or ""
        ).strip().upper()

        if finish_reason == "MAX_TOKENS":
            usage_raw = (
                data.get("usageMetadata") or {}
            )

            raise RuntimeError(
                (
                    "Google Gemini OCR was truncated "
                    "(finishReason=MAX_TOKENS). "
                    "No incomplete OCR bundle was saved. "
                    f"Configured maxOutputTokens="
                    f"{generation_config['maxOutputTokens']}; "
                    f"promptTokenCount="
                    f"{usage_raw.get('promptTokenCount')}; "
                    f"candidatesTokenCount="
                    f"{usage_raw.get('candidatesTokenCount')}; "
                    f"thoughtsTokenCount="
                    f"{usage_raw.get('thoughtsTokenCount')}. "
                    "Increase GOOGLE_OCR_MAX_OUTPUT_TOKENS or "
                    "GEMINI_OCR_MAX_OUTPUT_TOKENS."
                )
            )

        text = _extract_gemini_text(data)

        if not text.strip():
            raise RuntimeError(
                "Google Gemini OCR returned empty text. First 1000 chars of response:\n"
                + json.dumps(data, ensure_ascii=False)[:1000]
            )

        old_model = self.model
        self.model = model_to_use
        try:
            usage = self._record_usage(
                data=data,
                task="ocr",
                source_filename=file_path.name,
                input_kind=guess_input_kind(file_path),
            )
        finally:
            self.model = old_model

        mode = "pdf" if suffix == ".pdf" else "image"

        return OcrResponse(
            provider="google",
            model=model_to_use,
            input_kind=guess_input_kind(file_path),
            source_filename=file_path.name,
            source_path=str(file_path),
            text=text,
            pages=[
                OcrPage(
                    page_index=0,
                    page_number=1,
                    text=text,
                )
            ],
            provider_mode=mode,
            provider_document_id=None,
            provider_status=None,
            response_id=None,
            usage=usage,
            raw=data,
        )