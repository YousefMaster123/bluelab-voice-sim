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
    # MUST import silero here (module load = main thread). The Speechmatics plugin auto-loads silero
    # inside STT.__init__, which runs on a job WORKER thread, and LiveKit forbids registering plugins
    # off the main thread ("Plugins must be registered on the main thread"). Pre-importing on the
    # main thread makes that later lazy import a cached no-op, avoiding the crash.
    from livekit.plugins import silero as _silero  # noqa: F401
    from livekit.plugins import speechmatics as _speechmatics

    # The non-deprecated way to pin the acoustic model on the underlying speechmatics-rt config.
    # `OperatingPoint` is deprecated there in favor of `Model` (same "enhanced"/"standard" values).
    from speechmatics.rt import Model as _RTModel
    from speechmatics.voice import AdditionalVocabEntry as _VocabEntry
    from speechmatics.voice import EndOfUtteranceMode as _EOUMode
    from speechmatics.voice._client import VoiceAgentClient as _VoiceAgentClient
except ImportError:  # pragma: no cover - plugin not installed
    _speechmatics = None
    _RTModel = None
    _VocabEntry = None
    _EOUMode = None
    _VoiceAgentClient = None


if _VoiceAgentClient is not None:
    # ── VENDOR BUG WORKAROUND (speechmatics-voice 0.2.8) ────────────────────────────────────────
    # Every client emission (partials, finals, turn events) drains through ONE queue worker,
    # `VoiceAgentClient._run_stt_queue`, whose `except RuntimeError: return` (written for "event
    # loop closed" at shutdown) catches ANY RuntimeError — e.g. a collection mutated while two
    # finalization paths race — and permanently kills the worker, logged only at DEBUG. Observed
    # in production as a silently deaf call: engine connected, VAD heard speech, zero transcripts,
    # zero errors. This replacement keeps the vendor behavior byte-for-byte EXCEPT: a RuntimeError
    # while the event loop is still running is logged at ERROR and survived (one glitched emission
    # instead of a dead call); only a genuinely closed loop exits. Remove when upstream fixes the
    # handler (their repo is active; see 0.2.9 RCs).
    async def _resilient_run_stt_queue(self) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        while True:
            try:
                callback = await self._stt_message_queue.get()
                if asyncio.iscoroutine(callback):
                    await callback
                elif asyncio.iscoroutinefunction(callback):
                    await callback()
                elif callable(callback):
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await result
            except asyncio.CancelledError:
                return
            except RuntimeError:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    _log.error("stt_queue_loop_closed_exiting")
                    return
                _log.error("stt_queue_runtime_error_survived", exc_info=True)
            except Exception:  # noqa: BLE001 — mirror vendor catch-all, but at ERROR not WARNING
                _log.error("stt_queue_exception_survived", exc_info=True)

    _VoiceAgentClient._run_stt_queue = _resilient_run_stt_queue


# Speechmatics custom dictionary: `content` is the spelling we want back, `sounds_like` are
# pronunciations to match against, written the way the word is SAID rather than spelled — the
# engine's wrong guesses («عارفاها», «اليوم») show roughly what shape it was hearing.
#
# MUST be AdditionalVocabEntry objects, not plain dicts. `_prepare_config` returns the VoiceAgent
# config, and the client later reads `e.content` off every entry when converting to the rt config —
# dicts crash it with AttributeError, which surfaces as an unrecoverable stt_error that kills the
# call right after the greeting. (Learned the hard way; the test below now pins the type.)
#
# Keep this list SHORT. Speechmatics biases the model toward every entry, so unrelated audio starts
# getting pulled toward these spellings — a large dictionary makes recognition worse, not better.
# Only add nouns that actually appear in calls and actually get missed.
_VOCAB_TERMS: list[tuple[str, list[str]]] = [
    ("أليانز", ["اليانز", "أليانس", "اليانس", "الايانز"]),
    # Code-switched English. Measured on sim-891c19bd838d: «interested» was said twice, mid-phrase
    # between Arabic function words («...يعني interested في تأمين صحي», «حضرتك interested ولا»),
    # and produced NOTHING at any interim stage — not a mis-transcription, a silent drop. Note the
    # same call kept «I'm» verbatim and transliterated «Cairo» to «كايرو», so the bilingual pack
    # handles English inconsistently per word rather than not at all.
    # The sounds_like are Arabic-script because that is the script the engine is decoding into.
    ("interested", ["انترستد", "إنترستد", "انتريستد", "انترست"]),
    # Same silent-drop signature, measured on sim-17574fab105d: across a 6.5-minute call the
    # ONLY English that survived STT was «interested» (the entry above) and «collaborations»;
    # «policy» and «partnerships» appear in no final and no interim at all.
    # «policy» is the harder of the two — the Arabic insurance term is «بوليصة», so the engine
    # has a strong competing native candidate and resolves toward it before dropping. The
    # sounds_like below spell the ENGLISH pronunciation, not the Arabic term, on purpose.
    ("policy", ["بوليسي", "بولسي", "بوليسى", "باليسي"]),
    ("partnerships", ["بارتنرشيبس", "بارتنرشيب", "بارتنرشبس", "بارتنرشيبز"]),
]

_ADDITIONAL_VOCAB = (
    [_VocabEntry(content=content, sounds_like=sounds_like) for content, sounds_like in _VOCAB_TERMS]
    if _VocabEntry is not None
    else []
)


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
            # Custom dictionary for proper nouns the acoustic model has no reason to know. Observed
            # on a real call: «أليانز» was never recognized at ANY interim stage — the engine
            # substituted the nearest common words instead («عارفاها», then «اليوم»), and only got
            # it right on the third attempt. That is out-of-vocabulary behaviour, distinct from the
            # segmentation drops elsewhere in this module, and a custom dictionary is its fix.
            config.additional_vocab = _ADDITIONAL_VOCAB
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
            # FAST FINALS (the latency lever): with forced EOU off, the plugin VAD's finalize()
            # flushes the already-streamed text LOCALLY the moment the VAD hears end of speech
            # (~0.55s) instead of awaiting a server confirmation round-trip — finals land at
            # ~0.6-0.8s, BEFORE the 1.0s turn-commit floor, which is what gives preemptive
            # generation a real head start every turn. The engine's FIXED silence timer (0.8s
            # trigger set on the constructor) is the required pairing for this flag combo and
            # acts as the server-side cleanup/backstop. These two finalizers can race; the race
            # is survivable ONLY because of the _resilient_run_stt_queue patch above — do not
            # run this combination without it (observed failure: silently deaf calls).
            config.end_of_utterance_mode = _EOUMode.FIXED
            config.end_of_turn_config.use_forced_eou = False
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
    # The BUNDLE's stt_provider wins over the STT_MODEL env var, mirroring how tts_provider
    # already works. This is deliberate: STT_MODEL lives in the agent's LiveKit Cloud secrets, and
    # `lk agent update-secrets` cannot update an existing secret in place — the only way is
    # `--overwrite`, which WIPES every other secret (Anthropic/xAI/Fish keys, the HMAC secret).
    # Bundle-driven selection means the sim can switch engines with a Railway variable instead.
    # Symmetric on purpose: whichever provider the bundle names wins, in BOTH directions. An
    # earlier version only forced deepgram, so a bundle asking for speechmatics still got Deepgram
    # whenever the STT_MODEL secret happened to say deepgram/* — the env silently overrode the
    # bundle in one direction only, which is exactly the kind of asymmetry that makes a rollback
    # look like it did nothing.
    provider = rc.stt_provider.strip().lower()
    model = settings.stt_model
    if provider == "deepgram" and not model.startswith("deepgram/"):
        model = "deepgram/nova-3"
    elif provider == "speechmatics" and not model.startswith("speechmatics/"):
        model = "speechmatics/enhanced"
    use_deepgram = model.startswith("deepgram/") and bool(settings.deepgram_api_key)
    use_direct = (
        model.startswith("speechmatics/")
        and bool(settings.speechmatics_api_key)
        and _speechmatics is not None
    )
    path = "deepgram_plugin" if use_deepgram else ("direct_plugin" if use_direct else "inference")
    _log.info(
        "stt_build",
        model=model,
        provider=rc.stt_provider,
        language=rc.stt_language,
        operating_point=rc.stt_operating_point,
        path=path,
    )
    if use_deepgram:
        return _build_direct_deepgram(bundle, settings, model)
    if use_direct:
        return _build_direct_speechmatics(bundle, settings)
    return inference.STT(model=model, language=rc.stt_language)


# Proper nouns the engine has no reason to know, which carry the most meaning in a sales call —
# a mangled company name derails the LLM's reply far more than a mangled filler word.
#
# nova-3 uses Keyterm Prompting (plain strings), NOT the older weighted `keywords` — the plugin
# rejects `keywords` on nova-3 outright. Deepgram documents keyterm as English-only today, so
# treat this as best-effort: it is passed, and if the API ignores it for `ar` we lose nothing.
_DEEPGRAM_KEYTERMS: list[str] = ["أليانز", "بيكسل بوينت", "مريم", "يوسف"]

# Deepgram language per bundle language. CRITICAL: never "multi" for Arabic — measured on a real
# Egyptian recording, nova-3 `multi` mis-detected Spanish/Norwegian and returned pure nonsense
# ("Ay, amla eh", "drørte I møsterke f a I policy"), while the same audio with language="ar"
# transcribed correctly. `ar` also beat the Speechmatics ar_en bilingual pack, which silently
# DROPPED about a third of the words on that clip.
_DEEPGRAM_LANGUAGES: dict[str, str] = {
    "ar": "ar",
    "ar_en": "ar",  # the bilingual code the Speechmatics path uses; Deepgram wants plain `ar`
    "multi": "ar",  # defensive: never let "multi" reach Deepgram on an Arabic attempt
    "fr": "fr",
    "en": "en-US",
}


def _deepgram_language(bundle: RuntimeBundle) -> str:
    lang = bundle.runtime_config.stt_language.strip().lower().replace("-", "_")
    if mapped := _DEEPGRAM_LANGUAGES.get(lang):
        return mapped
    if mapped := _DEEPGRAM_LANGUAGES.get(lang.split("_", 1)[0]):
        return mapped
    return "ar" if bundle.is_arabic_or_mixed else "en-US"


def _build_direct_deepgram(bundle: RuntimeBundle, settings: Settings, stt_model: str) -> stt.STT:
    """Direct Deepgram nova-3. Measurably more accurate on Egyptian Arabic than Speechmatics.

    Trade-off, measured on the same 12.6s clip: nova-3's finals arrive ~1-3s LATER than
    Speechmatics', because it waits for more context — which is exactly why it keeps the words
    Speechmatics drops. If end-of-turn latency regresses, retune the turn detector's
    min/max_endpointing_delay, not this.
    """
    from livekit.plugins import deepgram

    model = stt_model.split("/", 1)[1] if "/" in stt_model else "nova-3"
    language = _deepgram_language(bundle)
    _log.info("deepgram_stt", model=model, language=language, keyterms=len(_DEEPGRAM_KEYTERMS))
    return deepgram.STT(
        api_key=settings.deepgram_api_key,
        model=model,
        language=language,
        interim_results=True,  # the turn detector + preemptive generation both consume interims
        # Deepgram's own endpointing stays minimal: turn-taking belongs to the session's
        # inference.TurnDetector, exactly as the Speechmatics path runs in EXTERNAL mode.
        no_delay=True,
        endpointing_ms=25,
        punctuate=True,
        smart_format=False,  # reformats numbers/dates; unwanted noise in a dialect transcript
        filler_words=True,  # "يعني"/"اه" are turn-taking signal, not junk
        keyterm=_DEEPGRAM_KEYTERMS,
        enable_diarization=False,  # single participant (the rep)
    )


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
        # No periods and no exclamation marks: with FIXED end-of-utterance, a mid-sentence pause
        # force-finalizes a fragment and the engine stamped each as a full sentence
        # («أنا. بكلمك من شركة.») — misleading boundaries that measurably confused the LLM. With
        # "." banned but "!" permitted, the engine SUBSTITUTED "!" as terminal punctuation on
        # every final («عاملة ايه!») — reads as constant emphasis, so ban it too. Statements end
        # unpunctuated (fine for a phone transcript the prompt already calls chopped); question
        # marks — the punctuation that changes meaning («معايا مريم؟») — stay permitted.
        punctuation_overrides={"sensitivity": 0.2, "permitted_marks": [",", "?"]},
        enable_diarization=False,  # single participant (the rep); no speaker split needed
        # EXTERNAL: the plugin's auto-loaded Silero VAD drives finalize() at end of speech; the
        # session's inference.TurnDetector owns turn-taking. Paired with the _prepare_config
        # override above (FIXED engine timer + forced EOU off) this is the fast-finals hybrid:
        # VAD-triggered INSTANT local flush (~0.6-0.8s finals) + engine silence timer as backstop.
        # Safe only with the resilient-queue patch at the top of this module. Do NOT use
        # SMART_TURN here. Native FIXED alone was tried and rejected: the engine's server-side
        # silence detection is noise-sensitive (finals drifted to 1.5-1.9s on long turns).
        turn_detection_mode=_speechmatics.TurnDetectionMode.EXTERNAL,
        end_of_utterance_silence_trigger=0.8,
        end_of_utterance_max_delay=1.6,
    )
