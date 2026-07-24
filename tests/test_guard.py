"""Tests-by-construction for the runtime-safe bundle + knowledge-boundary guard (AIR-2).

Runnable with `uv run pytest` from `packages/runtime-bundle`. No network/IO — pure schema +
guard behavior, so it doubles as an import smoke check.
"""

from __future__ import annotations

import pytest

from bluelab_runtime_bundle import (
    KnowledgeBoundaryError,
    PersonaSections,
    RuntimeBundle,
    assert_payload_safe,
    assert_runtime_safe,
    find_forbidden_fields,
)


def _valid_bundle() -> RuntimeBundle:
    return RuntimeBundle(
        attempt_id="att_123",
        org_id="org_456",
        livekit_room="room_att_123",
        livekit_session_id="sess_789",
        call_type="discovery",
        lead_type="referral",
        language="ar",
        wrapper_type="training",
        voice="eve",
        persona=PersonaSections(
            who_you_are="You are Mona, a cautious procurement lead who hates being upsold.",
            your_world="Mid-size logistics firm in Cairo; you already run a competitor policy.",
            where_you_are_right_now="Busy, mildly skeptical, took the call as a favor.",
            call_context="A mutual contact referred the rep; you know little about them.",
        ),
        prompt_versions={"voice_guardrails": "voice-guardrails@v7"},
        model_versions={"voice_agent": "claude-haiku-4-5"},
    )


def test_valid_bundle_passes_guard() -> None:
    bundle = _valid_bundle()
    assert assert_runtime_safe(bundle) is bundle
    assert bundle.is_arabic_or_mixed is True
    # Defaults resolved from the spec picks.
    assert bundle.runtime_config.llm_model == "claude-haiku-4-5"
    assert bundle.runtime_config.stt_provider == "speechmatics"
    assert bundle.runtime_config.tts_provider == "xai"


def test_persona_hidden_motive_prose_is_allowed() -> None:
    # The persona's own psychology (its hidden motive, in prose) is runtime-safe; only the
    # evaluator's *rubric* view of it is forbidden. A persona mentioning a hidden motive in
    # section 2A must NOT trip the guard.
    bundle = _valid_bundle()
    bundle.persona.who_you_are += " Your hidden motive: you fear looking incompetent to your boss."
    assert assert_runtime_safe(bundle) is bundle


@pytest.mark.parametrize(
    "forbidden_payload",
    [
        {"product_facts": [{"claim": "Plan A covers flood damage"}]},
        {"answer_key": "the rep should mention the 30-day window"},
        {"rubric": {"dimensions": []}},
        {"scorecard": {"overall_score": 80}},
        {"selected_product_snapshot": {"facts": []}},
        {"wrapper_references": {"candidate_id": "c_1"}},
        {"candidate_id": "c_1"},
        {"api_key": "sk-leak"},
        {"hmac_secret": "deadbeef"},
        # nested deep inside an otherwise-fine structure
        {"runtime_config": {"nested": {"rubric_dimensions": [1, 2, 3]}}},
        {"persona": {"who_you_are": "ok", "scoring": {"weight": 10}}},
    ],
)
def test_forbidden_payloads_are_rejected(forbidden_payload: dict) -> None:
    hits = find_forbidden_fields(forbidden_payload)
    assert hits, f"expected a forbidden hit for {forbidden_payload}"
    with pytest.raises(KnowledgeBoundaryError):
        assert_payload_safe(forbidden_payload)


def test_extra_top_level_key_fails_to_parse() -> None:
    # Defense-in-depth: extra="forbid" means a leaked key never even constructs.
    import pydantic

    data = _valid_bundle().model_dump(mode="json")
    data["product_facts"] = ["leaked"]
    with pytest.raises(pydantic.ValidationError):
        RuntimeBundle.model_validate(data)


def test_english_bundle_is_not_arabic_or_mixed() -> None:
    bundle = _valid_bundle()
    bundle.language = "en"
    assert bundle.is_arabic_or_mixed is False
