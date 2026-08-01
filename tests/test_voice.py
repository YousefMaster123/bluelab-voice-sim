"""In-process tests for the voice agent (needs the livekit-agents stack; no network).

Covers: prompt assembly stays knowledge-bounded + byte-stable Section 1, the STT/TTS builders
construct the right plugins, HMAC signing round-trips (new X-Agent-Signature scheme), and settings
carry no DB credentials. Requires the livekit deps installed; the contract-only checks that do NOT
need livekit live in test_contract.py.
"""

from __future__ import annotations

import pytest
from livekit.agents import inference
from livekit.plugins import xai

from bluelab_runtime_bundle import PersonaSections, RuntimeBundle, assert_runtime_safe
from bluelab_voice.agent import (
    GUARDRAILS_VERSION,
    STATIC_PROMPT_BLOCKS,
    ProspectAgent,
    build_system_prompt,
)
from bluelab_voice.config import Settings
from bluelab_voice.signing import SIGNATURE_HEADER, sign_body, verify_signature
from bluelab_voice.stt import build_stt
from bluelab_voice.tts import build_tts


def _bundle() -> RuntimeBundle:
    return RuntimeBundle(
        attempt_id="att_smoke",
        org_id="org_smoke",
        livekit_room="attempt_att_smoke",
        call_type="discovery",
        lead_type="cold_outreach",
        language="ar",
        wrapper_type="hiring",
        voice="leo",
        persona=PersonaSections(
            who_you_are="You are Tarek, a blunt SME owner who distrusts cold callers.",
            your_world="You run a 20-person trading company; you self-insure today.",
            where_you_are_right_now="Annoyed at the interruption, ready to hang up.",
            call_context="Cold outreach; you don't know this salesperson at all.",
        ),
        prompt_versions={"voice_guardrails": GUARDRAILS_VERSION},
        model_versions={"voice_agent": "claude-haiku-4-5"},
    )


def test_bundle_passes_guard() -> None:
    assert assert_runtime_safe(_bundle()) is not None


def test_system_prompt_contains_guardrails_and_sections() -> None:
    prompt = build_system_prompt(_bundle())
    for block in STATIC_PROMPT_BLOCKS:
        assert block in prompt
    assert prompt.startswith(STATIC_PROMPT_BLOCKS[0])  # the preamble frames everything
    assert "WHO YOU ARE" in prompt and "Tarek" in prompt
    assert "CALL CONTEXT" in prompt
    # v19 narrative order: the scene lands LAST (recency), with CALL CONTEXT just before it.
    # (rindex: the rules also MENTION these section names, so compare the headers themselves.)
    assert prompt.rindex("CALL CONTEXT") < prompt.rindex("WHERE YOU ARE RIGHT NOW")
    assert prompt.rstrip().endswith("Annoyed at the interruption, ready to hang up.")
    lowered = prompt.lower()
    for forbidden in ("rubric", "answer_key", "scorecard", "success_criteria"):
        assert forbidden not in lowered


def test_end_call_is_the_only_tool_and_is_described_to_the_model() -> None:
    """v22: the persona can hang up. One tool only — 07 §6 stays tool-light."""
    tools = ProspectAgent(_bundle()).tools
    assert [t.__name__ for t in tools] == ["end_call"]
    # The prompt must actually TELL it the tool exists and what earns a hangup, otherwise the
    # tool is dead weight (the pre-v22 state: the prompt promised a hangup with nothing behind it).
    prompt = build_system_prompt(_bundle())
    assert "end_call" in prompt and "<ending_the_call>" in prompt


def test_fish_provider_builds_fish_tts_and_requires_a_voice_id() -> None:
    """tts_provider=fishaudio routes to the direct Fish plugin, keyed by reference id."""
    from livekit.plugins import fishaudio

    bundle = _bundle()
    bundle.runtime_config.tts_provider = "fishaudio"
    bundle.runtime_config.tts_voice_id = "14f1000b77d547eeb5f03b474dd29e0f"
    bundle.runtime_config.tts_model = "s2.1-pro-free"
    bundle.runtime_config.tts_speed = 1.2
    engine = build_tts(bundle, Settings(fish_api_key="k"))
    assert isinstance(engine, fishaudio.TTS)

    # A Fish voice lives in runtime_config, NOT bundle.voice (still the leo/eve Literal) — so a
    # missing id must fail loudly rather than silently synthesizing with Fish's default voice.
    bundle.runtime_config.tts_voice_id = None
    with pytest.raises(ValueError, match="tts_voice_id"):
        build_tts(bundle, Settings(fish_api_key="k"))


def test_deepgram_is_the_default_stt_and_never_gets_multi_for_arabic() -> None:
    """nova-3 via the direct plugin, and `multi` must never reach it on an Arabic attempt.

    Measured on a real Egyptian recording: nova-3 with language="multi" mis-detected Spanish and
    Norwegian and returned nonsense, while language="ar" transcribed correctly. That makes the
    mapping below a correctness guard, not a preference.
    """
    from livekit.plugins import deepgram

    from bluelab_voice.stt import _deepgram_language

    assert Settings().stt_model == "deepgram/nova-3"

    bundle = _bundle()
    for stt_language in ("ar", "ar_en", "multi", "AR-EN"):
        bundle.runtime_config.stt_language = stt_language
        assert _deepgram_language(bundle) == "ar", stt_language

    bundle.runtime_config.stt_language = "ar_en"
    engine = build_stt(bundle, Settings(stt_model="deepgram/nova-3", deepgram_api_key="k"))
    assert isinstance(engine, deepgram.STT)


def test_static_blocks_are_byte_stable() -> None:
    for block in STATIC_PROMPT_BLOCKS:
        for token in ("{", "}", "%s", "format("):
            assert token not in block


def test_hmac_sign_and_verify_roundtrip() -> None:
    # New scheme: single X-Agent-Signature over the raw body, no timestamp.
    secret = "test-secret"
    body = b'{"attempt_id":"att_smoke","sequence_index":0}'
    headers = sign_body(secret, body)
    assert set(headers) == {SIGNATURE_HEADER}
    assert verify_signature(secret, body, headers[SIGNATURE_HEADER])
    assert not verify_signature(secret, b"tampered", headers[SIGNATURE_HEADER])


def test_settings_have_no_supabase_fields() -> None:
    fields = set(Settings.model_fields)
    for forbidden in ("supabase_url", "supabase_anon_key", "supabase_service_role_key"):
        assert forbidden not in fields


def test_stt_falls_back_to_inference_without_a_provider_key() -> None:
    # With NO provider key, STT falls back to LiveKit Inference; TTS uses the DIRECT xAI plugin
    # (needs XAI_API_KEY) — the network call is deferred to stream open, so a dummy key is enough.
    # The empty keys are explicit on purpose: Settings() reads .env, and a developer machine that
    # has real Speechmatics/Deepgram keys would otherwise take a direct path and fail this here.
    settings = Settings(xai_api_key="test-xai-key", speechmatics_api_key="", deepgram_api_key="")
    assert isinstance(build_stt(_bundle(), settings), inference.STT)
    assert isinstance(build_tts(_bundle(), settings), xai.TTS)


def test_settings_carry_the_direct_provider_keys() -> None:
    # The direct paths (Anthropic LLM + xAI TTS) each need their own key; STT is Inference (no key).
    fields = set(Settings.model_fields)
    assert "anthropic_api_key" in fields
    assert "xai_api_key" in fields


def test_stt_tts_models_are_config_driven() -> None:
    swapped = Settings(
        stt_model="deepgram/nova-3", xai_api_key="k", speechmatics_api_key="", deepgram_api_key=""
    )
    assert swapped.stt_model == "deepgram/nova-3"
    # No Deepgram key -> Inference serves the same model id (the key is what selects the plugin).
    assert isinstance(build_stt(_bundle(), swapped), inference.STT)
