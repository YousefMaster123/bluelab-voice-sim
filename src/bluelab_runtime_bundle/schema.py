"""The RuntimeBundle: the *only* material the Voice Prospect Agent is allowed to see.

This schema is the shared contract between two streams (REPO-3): `apps/api` **builds**
a bundle from the `attempt_snapshots → prospect_packages` reference (the allowlist
builder, AIR-2) and `apps/voice-agent` **validates** it before starting a roleplay
(REPO-4 — the worker imports only this package + the provider SDKs).

The knowledge boundary (SYS-3 / RE-7 / AIR-2) is expressed *structurally* here: the bundle
carries the four persona sections used verbatim, language/voice/runtime settings, and the
room/session identifiers the agent needs — and **nothing else**. There is deliberately no
field for product facts, the evaluator answer key, the rubric/scorecard, or the snapshot's
`selected_product_snapshot` (which is evaluator-only truth). `assert_runtime_safe` (see
`guard.py`) is the runtime enforcement that a constructed bundle has not smuggled any of
those in via `extra` keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Mirrors the engine value catalogs (01 §4 / 04 enums). Kept as Literals so a bad
# call_type/lead_type/language fails closed at the boundary rather than reaching the LLM.
CallType = Literal["discovery", "post_proposal", "renewal", "upsell"]
LeadType = Literal["inbound_quote", "referral", "cold_outreach"]
WrapperType = Literal["training", "hiring"]
# xAI Grok voices the spec pins (07 §1): leo (male) / eve (female).
Voice = Literal["leo", "eve"]
Gender = Literal["female", "male"]


class PersonaSections(BaseModel):
    """The four generated `prospect_packages` sections, used **verbatim** (RE-15 / AIR-5).

    These are sections 2A/2B/3/4. They are written in the third person and describe the
    prospect the agent embodies; Section 1 Guardrails are prepended from versioned code at
    prompt-assembly time (07 §2) and are NOT carried here. By RE-7 these sections never
    contain raw product knowledge — only what the persona would plausibly know.
    """

    model_config = ConfigDict(extra="forbid")

    who_you_are: str = Field(
        min_length=1,
        description="Section 2A — identity, personality, "
        "verbal habits, the hidden motive as the persona experiences it",
    )
    your_world: str = Field(
        min_length=1,
        description="Section 2B — the prospect's factual "
        "reality (company/household) and product-level facts a rep would probe",
    )
    where_you_are_right_now: str = Field(
        min_length=1, description="Section 3 — current state and opening energy"
    )
    call_context: str = Field(
        default="",
        description="Section 4 — the prospect's knowledge boundary; depth set by call type. "
        "Optional: leave empty when the persona needs no explicit call context (e.g. a cold call "
        "where the prospect knows nothing walking in). Omitted from the prompt when blank.",
    )


class RuntimeConfig(BaseModel):
    """Resolved turn-detection / VAD / interruption / STT-LLM-TTS settings (07 §6, VR-7).

    This is the runtime half of what `attempt_snapshots.runtime_config` captures, so an
    attempt reproduces. Per AIR-9 the concrete plugin/model IDs are verified against live
    LiveKit docs at build time; the defaults below are the spec's picks (Speechmatics STT,
    xAI Grok TTS, Claude Haiku LLM) with Arabic/mixed falling back to language-agnostic VAD.
    """

    model_config = ConfigDict(extra="forbid")

    # STT — Speechmatics bilingual; `ar` is the global model that handles Egyptian-Arabic↔
    # English code-switching (07 §1). The `ar_en` label in the spec is logical; the plugin
    # parameter is a single language code.
    stt_provider: str = "speechmatics"
    stt_language: str = "ar"
    stt_operating_point: str = "enhanced"

    # LLM — Claude Haiku, direct Anthropic path (AIR-1), thinking disabled for latency (07 §1),
    # prompt caching of the stable Section-1 prefix is the primary latency lever (AIR-5).
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 1024
    llm_prompt_caching: bool = True

    # TTS — xAI Grok; voice (leo/eve) is carried separately on the bundle.
    tts_provider: str = "xai"
    tts_model: str = "tts-1"

    # Turn detection — per-language; Arabic/mixed → language-agnostic VAD (07 §1/§6).
    turn_detection: Literal["vad", "stt", "multilingual"] = "vad"
    allow_interruptions: bool = True
    # Endpointing window targets (07 §9): silence → final.
    min_endpointing_delay: float = 0.4
    max_endpointing_delay: float = 6.0


class RuntimeBundle(BaseModel):
    """Runtime-safe material handed to the Voice Prospect Agent — the allowlist (AIR-2).

    Built by `apps/api` from the `attempt_snapshots → prospect_packages` reference and fetched
    by `apps/voice-agent` over signed service auth (09 §3). Validate with
    `assert_runtime_safe(bundle)` before use; the guard hard-rejects any forbidden field.

    `extra="forbid"` means a bundle that arrives with an unexpected key (e.g. a leaked
    `product_facts`) fails to even parse — defense-in-depth alongside the explicit guard.
    """

    model_config = ConfigDict(extra="forbid")

    # Schema/version marker so a worker can reject a bundle it can't safely interpret.
    bundle_version: Literal[1] = 1

    # Correlation + room/session refs (the only identifiers the agent needs — 01 §11 / 07 §5).
    attempt_id: str = Field(min_length=1, description="Correlation id; never a wrapper id")
    org_id: str = Field(min_length=1)
    livekit_room: str = Field(min_length=1)
    livekit_session_id: str | None = Field(default=None)

    # Call parameters (non-sensitive; these are the same IDs dispatch/room metadata may carry).
    call_type: CallType
    lead_type: LeadType | None = None
    language: str = Field(min_length=2, max_length=16)
    wrapper_type: WrapperType

    # The persona (verbatim sections) + speech identity.
    persona: PersonaSections
    voice: Voice
    # Explicit genders (optional). When BOTH are present the worker renders an explicit
    # <gender> prompt section from them; when absent it falls back to generic wording that
    # points at the persona sections. persona_gender = the human the agent plays;
    # caller_gender = the person on the line (the rep).
    persona_gender: Gender | None = None
    caller_gender: Gender | None = None

    # Resolved runtime config + the versions recorded on the snapshot (AIR-10) so the call
    # reproduces. These are *version strings*, never prompt bodies or answer keys.
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    prompt_versions: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'voice_guardrails': 'voice-guardrails@v7'} — version keys only",
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'voice_agent': 'claude-haiku-4-5'} — model ids only",
    )

    @property
    def is_arabic_or_mixed(self) -> bool:
        """True when the language requires the language-agnostic VAD path (07 §1/§6)."""
        lang = self.language.lower()
        return lang.startswith("ar") or lang in {"mixed", "ar-en", "ar_en", "multi"}
