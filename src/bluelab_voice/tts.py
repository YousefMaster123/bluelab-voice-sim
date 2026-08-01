"""xAI Grok TTS via the direct plugin (07 §1, AIR-9).

xAI Grok TTS is the pick for `ar-EG` (Egyptian) naturalness + expressive speech tags
([laugh], [pause], <emphasis>…) + sub-second streaming (07 §1). We call it through the **direct
xAI plugin** (`xai.TTS`, own API key) rather than LiveKit Inference: Inference proxies the request
through LiveKit's backend, which intermittently fails to reach xAI (DNS SERVFAIL on their infra);
the direct plugin calls `api.x.ai` straight from the worker. The spec pins voices `leo` / `eve`.

TTS streaming (first audio out before full generation) is half the latency budget (07 §8/§9).
"""

from __future__ import annotations

from livekit import agents
from livekit.agents import tokenize
from livekit.plugins import xai
from livekit.plugins.xai import tts as _xai_tts

from bluelab_runtime_bundle import RuntimeBundle

from .config import Settings
from .logging import get_logger

_log = get_logger("bluelab.voice.tts")

# QUALITY EXPERIMENT (retry): ask the streaming endpoint for 44.1 kHz (the xai plugin hardcodes 24 kHz
# with no constructor knob, so we patch its module constant). If xAI's streaming path ignores it and
# keeps sending 24 kHz, playback pitch goes wrong (slow/deep) — remove this line to revert.
_xai_tts.SAMPLE_RATE = 44100


# xAI TTS `language` hint per bundle language. xAI's accepted Arabic codes are `ar-EG`, `ar-SA`
# and `ar-AE` (verified against xAI's language list), so each market maps to its own code or the
# nearest accent: Qatari → `ar-AE` (lower-Gulf coastal dialect, closest to Emirati) and
# Kuwaiti → `ar-SA` (northern-Gulf dialect with Najdi roots, closest to Saudi). The dialect
# vocabulary itself comes from the system prompt, not this hint — this only nudges pronunciation.
_TTS_LANGUAGES: dict[str, str] = {
    "ar-eg": "ar-EG",
    "ar-sa": "ar-SA",
    "ar-ae": "ar-AE",
    "ar-qa": "ar-AE",
    "ar-kw": "ar-SA",
    "fr": "auto",
    "en": "auto",
}


def _tts_language(bundle: RuntimeBundle) -> str:
    """Map the bundle language to xAI's `language` hint (exact, then base subtag, then legacy)."""
    lang = bundle.language.strip().lower().replace("_", "-")
    if mapped := _TTS_LANGUAGES.get(lang):
        return mapped
    if mapped := _TTS_LANGUAGES.get(lang.split("-", 1)[0]):
        return mapped
    # Legacy/generic Arabic codes (`ar`, `ar_en`, `mixed`, `multi`) keep the Egyptian voice.
    return "ar-EG" if bundle.is_arabic_or_mixed else "auto"


def _build_fish_tts(bundle: RuntimeBundle, settings: Settings) -> agents.tts.TTS:
    """Fish Audio via the direct plugin — the only route to a custom/community voice.

    LiveKit Inference exposes Fish's default voice library only ("pre-existing custom Fish Audio
    voices are not available through LiveKit Inference"), so a voice picked from the playground
    requires our own key + this plugin.

    NOTE there is no `language` argument on this plugin (unlike xAI): with Fish the accent is a
    property of the trained voice, so `_tts_language` has nothing to feed it. A voice trained on
    Egyptian Arabic stays Egyptian in every market.
    """
    from livekit.plugins import fishaudio

    cfg = bundle.runtime_config
    voice_id = (cfg.tts_voice_id or "").strip()
    if not voice_id:
        raise ValueError("tts_provider=fishaudio requires runtime_config.tts_voice_id")

    _log.info(
        "tts_build",
        provider="fishaudio",
        model=cfg.tts_model,
        voice_id=voice_id,
        speed=cfg.tts_speed,
        language_hint="n/a (baked into the voice)",
    )
    kwargs: dict[str, object] = {}
    if cfg.tts_speed is not None:
        kwargs["speed"] = cfg.tts_speed
    return fishaudio.TTS(
        api_key=settings.fish_api_key or agents.NOT_GIVEN,
        # `model` is typed `TTSModels | str`, so ids outside the literal (e.g. the credit-free
        # `s2.1-pro-free`) pass straight through to the API.
        model=cfg.tts_model,
        voice_id=voice_id,
        # Match the xAI path's 44.1 kHz so switching provider doesn't also change audio quality.
        sample_rate=44100,
        # MEASURED, not assumed. This was "normal" ("favour quality over latency") until a
        # three-run-per-mode comparison on the production voice + Arabic text showed what that
        # actually cost: normal 1.37s mean time-to-first-byte vs balanced 0.62s vs low 0.65s. The
        # quality assumption was never tested and the penalty was ~0.75s on EVERY turn. `low` buys
        # nothing further on Arabic, so `balanced` (also the plugin's own default) it is.
        # Incidentally: Arabic synthesis is FASTER than English here, so the dialect was never
        # the latency problem it looked like.
        latency_mode="balanced",
        **kwargs,  # type: ignore[arg-type]
    )


def build_tts(bundle: RuntimeBundle, settings: Settings) -> agents.tts.TTS:
    """Construct the bundle's TTS: direct xAI Grok (leo/eve), or Fish Audio when pinned."""
    if bundle.runtime_config.tts_provider.strip().lower() in ("fish", "fishaudio"):
        return _build_fish_tts(bundle, settings)

    language = _tts_language(bundle)
    _log.info(
        "tts_build",
        provider=bundle.runtime_config.tts_provider,
        voice=bundle.voice,
        language=language,
    )
    return xai.TTS(
        api_key=settings.xai_api_key or agents.NOT_GIVEN,
        voice=bundle.voice,
        language=language,
        # 0 = xAI's highest-quality streaming mode (least latency-optimization). Trades a little
        # first-audio latency for better prosody, to get closer to the batch/portal render.
        optimize_streaming_latency=0,
        # CRITICAL FIX: the plugin's default WordTokenizer strips whitespace and the streaming path
        # concatenates each token with NO space — so xAI receives run-on text with no word
        # boundaries, wrecking pronunciation. retain_format=True keeps each token's leading space so
        # the reconstructed text matches the original exactly. (Verified against livekit-plugins-xai.)
        tokenizer=tokenize.basic.WordTokenizer(ignore_punctuation=False, retain_format=True),
    )
