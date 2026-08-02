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
from livekit.plugins import anthropic, openai

from bluelab_runtime_bundle import RuntimeBundle

from .config import Settings
from .logging import get_logger

_log = get_logger("bluelab.voice.llm")


def _build_openai_llm(bundle: RuntimeBundle, settings: Settings) -> openai.LLM:
    """OpenAI via the direct plugin — the default since the bake-off.

    Chosen on a 7-model replay of a real call (model_comparisons/): gpt-5.3-chat-latest
    produced the most idiomatic Egyptian of any model tested (native constructions like
    «بقالنا شوية» / «داخل على تجديد» that no other model reached for), at 0.81s median
    TTFT vs 1.32s for claude-sonnet-4-5, and 51 median output tokens vs 77 — which
    matters as much as latency, because every output token is also TTS speaking time.

    The `-chat-latest` line is deliberate: it is NON-reasoning, so there is no reasoning
    budget to disable and nothing that can silently re-enable it and turn a 0.8s turn
    into a 15s one. Do not "upgrade" this to a bare gpt-5.x id without re-checking that.

    CACHING WORKS DIFFERENTLY FROM ANTHROPIC. There is no cache_control breakpoint —
    OpenAI caches any prefix over ~1024 tokens automatically (our prompt is ~4.2k, well
    clear). What we control is HIT RATE:
      * `prompt_cache_key` routes requests with the same key to the same machine. It is
        keyed on the guardrails version + language, NOT the attempt id — an attempt-scoped
        key would give every call a cold cache, which is the opposite of the point.
      * `prompt_cache_retention="24h"` keeps the entry alive between calls. The default
        in-memory cache expires in minutes, and sim traffic is sporadic, so without this
        nearly every call pays full price for the prefix.
    """
    # Imported here, not at module scope: agent.py imports build_llm from this module, so a
    # top-level import would be circular.
    from .agent import GUARDRAILS_VERSION

    rc = bundle.runtime_config
    cache_key = f"bluelab-{GUARDRAILS_VERSION}-{bundle.language}"
    _log.info(
        "llm_build",
        provider="openai",
        model=rc.llm_model,
        caching=bool(rc.llm_prompt_caching),
        cache_key=cache_key,
    )
    kwargs: dict = {}
    if rc.llm_prompt_caching:
        kwargs["prompt_cache_key"] = cache_key
        kwargs["prompt_cache_retention"] = "24h"
    return openai.LLM(
        model=rc.llm_model,
        api_key=settings.openai_api_key or agents.NOT_GIVEN,
        temperature=rc.llm_temperature,
        max_completion_tokens=rc.llm_max_tokens,
        # Same budget as the Anthropic path: generous read for first token under load.
        timeout=httpx.Timeout(5.0, read=30.0),
        **kwargs,
    )


def build_llm(bundle: RuntimeBundle, settings: Settings) -> agents.llm.LLM:
    """Construct the bundle's LLM: OpenAI by default, Anthropic when the bundle says so."""
    if bundle.runtime_config.llm_provider.strip().lower() == "openai":
        return _build_openai_llm(bundle, settings)
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
