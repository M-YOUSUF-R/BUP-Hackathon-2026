# app/llm_health.py
"""
Cached LLM health probe.

Runs a tiny request against the configured provider and classifies failures:
  - auth        : missing or invalid API key
  - rate_limit  : quota exhausted or 429
  - token_limit : context/token limit exceeded
  - not_found   : model not available at that provider
  - timeout     : probe timed out
  - network     : provider unreachable
  - bad_request : provider rejected the request (400)
  - unknown     : anything else

The result is cached for LLM_HEALTH_CACHE_SECONDS so repeated /health polls
don't burn tokens or add latency.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = float(os.getenv("LLM_HEALTH_CACHE_SECONDS", "30"))
_PROBE_TIMEOUT = float(os.getenv("LLM_HEALTH_TIMEOUT_SECONDS", "6"))


@dataclass
class LLMHealth:
    ok: bool = False
    provider: str = "unknown"
    model: str = "unknown"
    latency_ms: Optional[int] = None
    error_type: Optional[str] = None
    message: str = ""
    checked_at: float = 0.0
    cached: bool = False


_cache: LLMHealth | None = None


def _detect_provider() -> str:
    base = (os.getenv("LLM_BASE_URL") or "").lower()
    if not base:
        return "openai"
    if "groq" in base:
        return "groq"
    if "together" in base:
        return "together"
    if "anthropic" in base:
        return "anthropic"
    if "ollama" in base or ":11434" in base:
        return "ollama"
    if "openrouter" in base:
        return "openrouter"
    if "azure" in base:
        return "azure-openai"
    return "custom"


def _classify_exception(exc: Exception) -> tuple[str, str]:
    """Map provider exceptions into a short category + human message."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "authentication" in name or "401" in msg or "invalid api key" in msg:
        return "auth", "LLM API key invalid, expired, or missing."
    if "ratelimit" in name or "429" in msg or "insufficient_quota" in msg or "quota" in msg:
        return "rate_limit", "LLM rate limit or quota exceeded."
    if ("context" in msg and "length" in msg) or ("token" in msg and "limit" in msg):
        return "token_limit", "LLM token/context limit exceeded."
    if "notfound" in name or "404" in msg or "model_not_found" in msg or "does not exist" in msg:
        return "not_found", "LLM model not found or unavailable at provider."
    if "timeout" in name or "timed out" in msg:
        return "timeout", "LLM request timed out."
    if "connection" in name or "network" in msg or "getaddrinfo" in msg or "unreachable" in msg:
        return "network", "Cannot reach LLM provider."
    if "badrequest" in name or "400" in msg:
        return "bad_request", "LLM rejected the probe request."
    return "unknown", f"LLM error: {type(exc).__name__}"


def _probe_llm() -> LLMHealth:
    """One cheap call. Never raises."""
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    provider = _detect_provider()
    base_url = os.getenv("LLM_BASE_URL")

    result = LLMHealth(
        ok=False,
        provider=provider,
        model=model,
        checked_at=time.time(),
    )

    # Local models often don't need a key. Hosted providers do.
    if provider != "ollama" and not os.getenv("OPENAI_API_KEY"):
        result.error_type = "auth"
        result.message = "LLM API key is not configured."
        return result

    t0 = time.perf_counter()
    try:
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            temperature=0,
            timeout=_PROBE_TIMEOUT,
            max_retries=0,
        )
        # Minimal request; output discarded. Costs a handful of tokens.
        llm.invoke("ok")
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        result.ok = True
        result.message = "LLM reachable."
    except Exception as exc:  # noqa: BLE001
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        result.error_type, result.message = _classify_exception(exc)
        logger.warning("LLM health probe failed: %s", exc)

    return result


def get_llm_health(force: bool = False) -> LLMHealth:
    """Return a fresh or cached LLMHealth. Never raises."""
    global _cache
    try:
        now = time.time()
        if (
            not force
            and _cache is not None
            and (now - _cache.checked_at) < _CACHE_TTL_SECONDS
        ):
            snapshot = LLMHealth(**vars(_cache))
            snapshot.cached = True
            return snapshot
        _cache = _probe_llm()
        return _cache
    except Exception as exc:  # noqa: BLE001
        # A bug in the probe must never break /health.
        logger.exception("LLM health probe crashed")
        return LLMHealth(
            ok=False,
            provider=_detect_provider(),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            error_type="unknown",
            message=f"Probe internal error: {type(exc).__name__}",
            checked_at=time.time(),
        )