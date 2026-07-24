"""Claude (Haiku) LLM adapter — the Voice Prospect Agent's brain (07 §1, AIR-1/AIR-5).

The agent runs on a **direct Anthropic Claude path** (`claude-haiku-4-5`), not LiveKit Inference
(Claude isn't in the Inference catalog — AIR-1 / 05 §11). Thinking is disabled: Haiku 4.5 doesn't
support `effort` and isn't a 4.6+ adaptive-thinking model, and live calls are latency-critical
(07 §1). Zero tools — the prospect is a single-phase roleplay (07 §6), so we never attach tools.

Prompt caching (AIR-5) is the single biggest latency lever: the Anthropic plugin's
`caching="ephemeral"` adds a `cache_control` breakpoint so the stable Section-1 guardrail prefix is
read at ~0.1× cost on repeat turns. NB: on Haiku the minimum cacheable prefix is 4096 tokens — if
the guardrails are shorter the cache silently won't engage (the guardrail author owns that, 07 §2).

Timeout + retry/degradation (07 §10): the plugin takes an httpx timeout (connect + read); on Claude
overload the plugin/SDK retry with backoff, and the worker's fallback is to end the turn gracefully.

Verified against docs.livekit.io (Anthropic plugin reference) at implementation time:
`anthropic.LLM` accepts `model`, `temperature`, `max_tokens`, `caching`, and `timeout`.
"""

from __future__ import annotations

import httpx
from livekit import agents
from livekit.plugins import anthropic

from bluelab_runtime_bundle import RuntimeBundle

from .config import Settings
from .logging import get_logger

_log = get_logger("bluelab.voice.llm")


def build_llm(bundle: RuntimeBundle, settings: Settings) -> anthropic.LLM:
    """Construct the Claude Haiku LLM plugin for the prospect agent (direct Anthropic, cached)."""
    rc = bundle.runtime_config
    caching = "ephemeral" if rc.llm_prompt_caching else agents.NOT_GIVEN
    _log.info(
        "llm_build",
        provider=rc.llm_provider,
        model=rc.llm_model,
        caching=bool(rc.llm_prompt_caching),
    )
    return anthropic.LLM(
        model=rc.llm_model,
        api_key=settings.anthropic_api_key or agents.NOT_GIVEN,
        temperature=rc.llm_temperature,
        max_tokens=rc.llm_max_tokens,
        caching=caching,
        # connect 5s / read 30s — generous read for first-token under load; no thinking budget.
        timeout=httpx.Timeout(5.0, read=30.0),
    )
