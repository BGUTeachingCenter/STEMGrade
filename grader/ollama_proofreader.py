from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional


# ---------------------------
# Public API
# ---------------------------

@dataclass
class ProofreadResult:
    tex: str
    used_ai: bool
    ok: bool
    reason: str
    meta: Dict[str, Any]


def proofread_tex_with_ollama(
    tex: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 120,
    retries: int = 1,
    num_predict: int = 1800,
    max_chars: int = 40_000,
    temperature: float = 0.0,
    safe_mode: bool = True,
) -> ProofreadResult:
    """
    Best-effort TeX proofread/fix using Ollama chat API.

    - Never raises on network/model errors (returns original tex).
    - If safe_mode=True: rejects outputs that look structurally worse.
    """
    original = tex

    # Gate: allow easy disable via env
    if os.getenv("MATHGRADE_OLLAMA_PROOFREAD", "").strip() in ("0", "false", "False", "no", "NO"):
        return ProofreadResult(
            tex=original, used_ai=False, ok=True, reason="disabled_by_env",
            meta={"env": "MATHGRADE_OLLAMA_PROOFREAD"}
        )

    model = model or os.getenv("MATHGRADE_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL") or "gemma3:4b"
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"

    # Keep input bounded
    clipped = _clip_tex_for_prompt(original, max_chars=max_chars)
    system_prompt = _system_prompt()
    user_prompt = _user_prompt(clipped)

    # Pre-structure metrics (for safe_mode)
    before_metrics = _structure_metrics(original) if safe_mode else {}

    try:
        reply = _ollama_chat(
            base_url=base_url,
            model=model,
            system=system_prompt,
            user=user_prompt,
            timeout_s=timeout_s,
            retries=retries,
            num_predict=num_predict,
            temperature=temperature,
        )
    except Exception as e:
        return ProofreadResult(
            tex=original,
            used_ai=False,
            ok=True,
            reason="ollama_failed",
            meta={"error": str(e), "base_url": base_url, "model": model},
        )

    candidate = _extract_tex_only(reply)

    if not candidate.strip():
        return ProofreadResult(
            tex=original,
            used_ai=False,
            ok=True,
            reason="empty_model_output",
            meta={"base_url": base_url, "model": model},
        )

    # Optional safety checks (don’t accept output that looks more broken)
    if safe_mode:
        after_metrics = _structure_metrics(candidate)
        ok, why = _is_output_acceptable(before_metrics, after_metrics)
        if not ok:
            return ProofreadResult(
                tex=original,
                used_ai=True,
                ok=False,
                reason=f"rejected_output:{why}",
                meta={
                    "base_url": base_url,
                    "model": model,
                    "before": before_metrics,
                    "after": after_metrics,
                },
            )

    return ProofreadResult(
        tex=candidate,
        used_ai=True,
        ok=True,
        reason="ok",
        meta={"base_url": base_url, "model": model, "clipped": clipped != original},
    )


# ---------------------------
# Ollama chat client
# ---------------------------

def _ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout_s: int,
    retries: int,
    num_predict: int,
    temperature: float,
) -> str:
    """
    Calls Ollama /api/chat with stream=False and returns assistant content.
    Raises on final failure (caller catches).
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            out = json.loads(raw)
            msg = out.get("message", {}) or {}
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content or "")
            return content.strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Ollama request failed: {e}") from e

    raise RuntimeError(f"Ollama request failed: {last_err}")


# ---------------------------
# Prompts
# ---------------------------

def _system_prompt() -> str:
    # Keep this short, strict, and “global”.
    return (
        "You are a LaTeX/XeLaTeX build fixer.\n"
        "Rules:\n"
        "- Make MINIMAL edits necessary to compile.\n"
        "- Preserve meaning; do not rewrite content.\n"
        "- Do NOT add explanations, comments, or markdown.\n"
        "- Output ONLY LaTeX source.\n"
        "- Do not invent packages unless necessary for compilation.\n"
        "- Keep existing structure; do not remove \\documentclass/\\begin{document} if present.\n"
    )


def _user_prompt(tex: str) -> str:
    # Task + payload goes in user message.
    return (
        "Fix the following LaTeX so it compiles with XeLaTeX.\n"
        "Return ONLY the corrected LaTeX source.\n"
        "If something is malformed (unbalanced braces/environments, bad commands), fix it.\n"
        "Here is the LaTeX:\n\n"
        "<<<LATEX>>>\n"
        f"{tex}\n"
        "<<<END_LATEX>>>\n"
    )


# ---------------------------
# Output post-processing
# ---------------------------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:latex|tex)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def _extract_tex_only(reply: str) -> str:
    """
    Removes code fences and trims obvious non-TeX wrappers.
    """
    s = reply.strip()
    s = _CODE_FENCE_RE.sub("", s).strip()

    # If model echoed markers, strip them
    s = re.sub(r"(?s)^.*<<<LATEX>>>\s*", "", s).strip()
    s = re.sub(r"(?s)\s*<<<END_LATEX>>>.*$", "", s).strip()

    return s


def _clip_tex_for_prompt(tex: str, *, max_chars: int) -> str:
    t = tex
    if len(t) <= max_chars:
        return t

    # Keep head + tail: errors are often at boundaries (preamble / end)
    head = t[: max_chars // 2]
    tail = t[-(max_chars // 2) :]
    return (
        head
        + "\n\n% --- [CLIPPED MIDDLE FOR LENGTH] ---\n\n"
        + tail
    )


# ---------------------------
# Safety checks (structure)
# ---------------------------

_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")

def _structure_metrics(tex: str) -> Dict[str, Any]:
    begins = _BEGIN_RE.findall(tex)
    ends = _END_RE.findall(tex)

    # Count brace imbalance (rough but useful)
    open_braces = tex.count("{")
    close_braces = tex.count("}")
    brace_delta = open_braces - close_braces

    # Track env mismatches
    begin_counts: Dict[str, int] = {}
    end_counts: Dict[str, int] = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1

    # Total mismatch magnitude
    env_delta = 0
    all_envs = set(begin_counts) | set(end_counts)
    for env in all_envs:
        env_delta += abs(begin_counts.get(env, 0) - end_counts.get(env, 0))

    has_docclass = "\\documentclass" in tex
    has_begin_doc = "\\begin{document}" in tex
    has_end_doc = "\\end{document}" in tex

    return {
        "len": len(tex),
        "brace_delta": brace_delta,
        "env_delta": env_delta,
        "has_docclass": has_docclass,
        "has_begin_doc": has_begin_doc,
        "has_end_doc": has_end_doc,
        "begin_counts": begin_counts,
        "end_counts": end_counts,
    }


def _is_output_acceptable(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Reject if it looks structurally worse than input.
    """
    # If input had a full doc structure, output shouldn’t delete it.
    if before.get("has_docclass") and not after.get("has_docclass"):
        return False, "lost_documentclass"
    if before.get("has_begin_doc") and not after.get("has_begin_doc"):
        return False, "lost_begin_document"
    if before.get("has_end_doc") and not after.get("has_end_doc"):
        return False, "lost_end_document"

    # Brace imbalance: allow improvement or same; reject if it increases in magnitude a lot.
    if abs(after.get("brace_delta", 0)) > abs(before.get("brace_delta", 0)) + 2:
        return False, "brace_imbalance_worse"

    # Environment mismatch: reject if significantly worse
    if after.get("env_delta", 0) > before.get("env_delta", 0) + 2:
        return False, "env_mismatch_worse"

    return True, "ok"
