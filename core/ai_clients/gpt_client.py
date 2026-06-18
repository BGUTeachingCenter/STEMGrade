# grader/ai_clients/gpt_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ssl

import requests
from requests.adapters import HTTPAdapter

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe_json_loads(s: str) -> dict:
    """
    - strips control chars
    - extracts the first {...} block if extra text exists
    """
    if not s:
        raise ValueError("Empty model response (expected JSON).")

    s2 = _CONTROL_CHARS_RE.sub("", s)

    # Some models might accidentally wrap JSON with text.
    start = s2.find("{")
    end = s2.rfind("}")
    candidate = s2[start : end + 1] if (start != -1 and end != -1 and end > start) else s2

    return json.loads(candidate)


def _extract_text_from_responses_api(resp_json: dict) -> str:
    """
    Responses API returns an 'output' array.
    We look for the first content item of type 'output_text' and return its 'text'.
    """
    out = resp_json.get("output")
    if isinstance(out, list):
        for item in out:
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        return c["text"]

    # Some responses include output_text directly
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"]

    return ""


def _get_ca_bundle_path() -> str:
    """
    Central place for certificate bundle discovery.

    For university / institutional proxies, set one of these env vars:
      SSL_CERT_FILE=C:\\Users\\alinag\\certs\\combined-ca-bundle.pem
      REQUESTS_CA_BUNDLE=C:\\Users\\alinag\\certs\\combined-ca-bundle.pem
      CURL_CA_BUNDLE=C:\\Users\\alinag\\certs\\combined-ca-bundle.pem
    """
    return (
        os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or ""
    ).strip()


class _RelaxedUniversitySSLAdapter(HTTPAdapter):
    """
    Requests adapter that keeps SSL verification ON, but relaxes Python/OpenSSL's
    strict X.509 validation.

    This is needed for some university proxy/root certificates that fail with:
      certificate verify failed: Missing Authority Key Identifier

    This is NOT the same as verify=False.
    """

    def __init__(self, ca_bundle: str, *args, **kwargs):
        self.ca_bundle = ca_bundle
        self.ssl_context = self._make_ssl_context(ca_bundle)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _make_ssl_context(ca_bundle: str) -> ssl.SSLContext:
        ctx = ssl.create_default_context(cafile=ca_bundle)

        # Keep certificate verification ON, but relax the extra strict
        # certificate-format checks introduced/enforced by newer Python/OpenSSL.
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT

        return ctx

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(
            connections,
            maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self.ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _make_requests_session() -> requests.Session:
    """
    Creates a requests session for OpenAI calls.

    If a CA bundle env var exists, use it and relax only the strict X.509 check
    that breaks with the university certificate. Otherwise, use normal requests.
    """
    session = requests.Session()

    ca_bundle = _get_ca_bundle_path()
    if ca_bundle:
        if not os.path.exists(ca_bundle):
            raise RuntimeError(
                "Certificate bundle path was set but does not exist:\n"
                f"{ca_bundle}\n\n"
                "Check SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE."
            )

        adapter = _RelaxedUniversitySSLAdapter(ca_bundle)
        session.mount("https://", adapter)

    return session


@dataclass
class GptClient:
    api_key: str
    model: str
    base_url: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # env reading ONLY here (mirrors your other clients)
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment (or pass api_key=...).")

        # Override via OPENAI_MODEL if you want
        self.model = (model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        self.session = _make_requests_session()

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: Dict[str, Any],
        temperature: float = 0.15,  # kept for compatibility with caller, but NOT sent
        timeout_s: int = 120,
        schema_name: str = "result",
        strict: bool = True,
    ) -> dict:
        """
        Returns a Python dict parsed from the model's structured JSON output.

        Uses the Responses API with Structured Outputs via:
          text.format = { type: "json_schema", ... }

        IMPORTANT:
          We intentionally do NOT send temperature/top_p because some models reject them.
          For grading, deterministic behavior is preferred anyway.
        """
        url = f"{self.base_url}/v1/responses"

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": bool(strict),
                    "schema": schema,
                }
            },
            # You can optionally add: "store": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        r = self.session.post(url, json=payload, headers=headers, timeout=timeout_s)

        if not r.ok:
            try:
                err = r.json()
                err_text = json.dumps(err, ensure_ascii=False, indent=2)
            except Exception:
                err_text = r.text[:2000]
            raise RuntimeError(f"OpenAI request failed ({r.status_code}). Error:\n{err_text}")

        data = r.json()

        text = _extract_text_from_responses_api(data)
        if not text:
            raise RuntimeError(
                "OpenAI returned no output_text. Response (trimmed):\n"
                + json.dumps(data, ensure_ascii=False, indent=2)[:2000]
            )

        try:
            return _safe_json_loads(text)
        except Exception as e:
            raise RuntimeError(
                "OpenAI returned non-JSON content. First 800 chars:\n"
                + text[:800]
            ) from e