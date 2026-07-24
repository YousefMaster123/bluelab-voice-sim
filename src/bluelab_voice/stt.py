"""Speechmatics STT via LiveKit Inference (07 §1, AIR-9).

The core requirement is Egyptian-Arabic↔English **intra-sentence code-switching** in one stream
(07 §1) — Deepgram nova-3 multilingual excludes Arabic, so Speechmatics is the pick. We reach it
through **LiveKit Inference** (`inference.STT`), so no Speechmatics API key is needed — usage is
billed through LiveKit Cloud and the model is swappable via `STT_MODEL` (the 07 §1 bake-off can
re-point it without code changes). The `language` parameter is a single global code; we use `ar`
(Speechmatics' global model recognizes dialects/accents incl. code-switching). The spec's `ar_en`
label is logical, not a literal param.

Turn detection is **VAD-based at the AgentSession level** (the prewarmed Silero VAD passed to the
session), per 07 §6 for the Arabic/mixed path — the Inference STT does not own endpointing, so the
plugin-only `turn_detection_mode`/`vad`/`operating_point` knobs no longer apply here (operating
point is encoded in the model id, e.g. `speechmatics/enhanced`).

> AIR-9 is normative: actual code-switching quality on real Egyptian audio is a first-class
> selection gate validated by a bake-off **before commit** — this adapter is the wiring, not the
> proof. Backup provider per 07 §1: Gladia (swap via `STT_MODEL`).

Verified against the LiveKit Agents SDK: `inference.STT(model=..., language=...)` is the Inference
STT constructor, and `speechmatics/enhanced` / `speechmatics/standard` are the exact ids in the
SDK's `SpeechmaticsModels` literal (`livekit/agents/inference/stt.py`). `enhanced` is the
higher-accuracy operating point; swap to `speechmatics/standard` (cheaper) via `STT_MODEL`.
"""

from __future__ import annotations

from livekit.agents import inference, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from bluelab_runtime_bundle import RuntimeBundle

from .config import Settings
from .logging import get_logger

_log = get_logger("bluelab.voice.stt")

# The direct Speechmatics plugin is optional (only needed for the bilingual ar_en path). Import it
# lazily-guarded so the Inference-only path keeps working without the plugin installed.
try:
    from livekit.plugins import speechmatics as _speechmatics

    # MUST import silero here (module load = main thread). The Speechmatics plugin auto-loads silero
    # inside STT.__init__, which runs on a job WORKER thread, and LiveKit forbids registering plugins
    # off the main thread ("Plugins must be registered on the main thread"). Pre-importing on the
    # main thread makes that later lazy import a cached no-op, avoiding the crash.
    from livekit.plugins import silero as _silero  # noqa: F401

    # The non-deprecated way to pin the acoustic model on the underlying speechmatics-rt config.
    # `OperatingPoint` is deprecated there in favor of `Model` (same "enhanced"/"standard" values).
    from speechmatics.rt import Model as _RTModel
except ImportError:  # pragma: no cover - plugin not installed
    _speechmatics = None
    _RTModel = None


if _speechmatics is not None:

    class _RawLanguageSpeechmaticsSTT(_speechmatics.STT):
        """Speechmatics STT that sends the language string to Speechmatics VERBATIM.

        The stock plugin wraps ``language`` in LiveKit's ``LanguageCode``, which rewrites the
        Speechmatics bilingual pack code ``ar_en`` -> ``ar-EN`` (a BCP-47 region form Speechmatics
        reads as Arabic-only, dropping English). We override ``_prepare_config`` to force the raw
        code so Speechmatics engages the real Arabic-English bilingual model. Verified: the stock
        ``_prepare_config`` sets ``config.language = LanguageCode(language)``; we overwrite it after.
        """

        def __init__(self, *args: object, raw_language: str, **kwargs: object) -> None:
            self._raw_language = raw_language
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN):  # type: ignore[override]
            config = super()._prepare_config(language)
            config.language = self._raw_language  # bypass LanguageCode normalization
            # Migrate operating_point -> model to clear the speechmatics-rt deprecation warning
            # ("TranscriptionConfig.operating_point is deprecated ... use the model property").
            # The speechmatics-voice client ALWAYS forwards VoiceAgentConfig.operating_point (which
            # defaults to ENHANCED and is never None) into the rt TranscriptionConfig, which is what
            # emits the warning — and it will be removed in a future release. The plugin exposes no
            # `model=` kwarg and VoiceAgentConfig has no `model` field, so we null operating_point
            # and inject the equivalent rt `model` via advanced_engine_control (the SDK's documented
            # setattr-merge onto the rt config). Preserves the standard/enhanced choice; touches only
            # the acoustic-model selection, nothing about turn-taking/endpointing.
            op = config.operating_point
            model = (
                _RTModel.STANDARD
                if (op is not None and str(op.value) == "standard")
                else _RTModel.ENHANCED
            )
            config.operating_point = None
            merged = dict(config.advanced_engine_control or {})
            merged["model"] = model
            config.advanced_engine_control = merged
            return config


def build_stt(bundle: RuntimeBundle, settings: Settings) -> stt.STT:
    """Construct the STT for this attempt.

    Two paths:
      * DIRECT Speechmatics plugin — used when ``stt_model`` is ``speechmatics/*`` AND a
        ``SPEECHMATICS_API_KEY`` is set. This is the ONLY path that can send the raw bilingual
        ``ar_en`` code (LiveKit Inference normalizes ``ar_en`` -> ``ar-EN`` = Arabic-only).
      * LiveKit Inference — everything else (no key, or a non-Speechmatics model). No provider key.

    Turn detection stays with the session's audio ``inference.TurnDetector``; the direct plugin runs
    in ``EXTERNAL`` turn-detection mode so it only drives transcript finalization, not endpointing.
    """
    rc = bundle.runtime_config
    use_direct = (
        settings.stt_model.startswith("speechmatics/")
        and bool(settings.speechmatics_api_key)
        and _speechmatics is not None
    )
    _log.info(
        "stt_build",
        model=settings.stt_model,
        provider=rc.stt_provider,
        language=rc.stt_language,
        operating_point=rc.stt_operating_point,
        path="direct_plugin" if use_direct else "inference",
    )
    if use_direct:
        return _build_direct_speechmatics(bundle, settings)
    return inference.STT(model=settings.stt_model, language=rc.stt_language)


def _build_direct_speechmatics(bundle: RuntimeBundle, settings: Settings) -> stt.STT:
    """Direct Speechmatics plugin with the raw language (bilingual ar_en) + tighter finalization."""
    from speechmatics.voice import OperatingPoint

    rc = bundle.runtime_config
    operating_point = (
        OperatingPoint.STANDARD
        if settings.stt_model.endswith("/standard")
        else OperatingPoint.ENHANCED
    )
    return _RawLanguageSpeechmaticsSTT(
        api_key=settings.speechmatics_api_key,
        raw_language=rc.stt_language,  # e.g. "ar_en" — sent verbatim, engages the bilingual model
        operating_point=operating_point,
        # max_delay 0.7 → 3.0: 0.7 finalized a chunk on every micro-pause and punctuated each as a
        # full sentence, so a question like «معايا مريم؟» arrived as «معايا. مريم.» — the LLM reads
        # that as a statement and greets the caller as "مريم". More delay = more context per chunk =
        # far saner punctuation. (Turn commit is VAD-driven in EXTERNAL mode, so end-of-turn latency
        # barely moves; min_delay=1.0 on the turn detector still matters.)
        max_delay=3.0,
        # Directly damp the false periods: much less trigger-happy with sentence-ending punctuation.
        punctuation_overrides={"sensitivity": 0.2},
        enable_diarization=False,  # single participant (the rep); no speaker split needed
        # EXTERNAL: the plugin only finalizes transcripts (via an auto-loaded Silero VAD); the
        # session's inference.TurnDetector still owns turn-taking. Do NOT use SMART_TURN here.
        turn_detection_mode=_speechmatics.TurnDetectionMode.EXTERNAL,
    )
