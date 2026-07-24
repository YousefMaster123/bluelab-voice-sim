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


# xAI TTS `language` hint per bundle language. `ar-EG` is the verified-good Egyptian voice, so the
# Egyptian/legacy-Arabic path keeps it exactly as before. The other dialects are sent as `auto` (xAI
# detects Arabic/French from the text itself) because their region codes are NOT verified against
# xAI's accepted list, and an unknown code fails the whole call rather than degrading. The dialect
# itself comes from the system prompt, not this hint — this only nudges pronunciation. If a region
# code is ever confirmed working, putting it here is the only edit needed.
_TTS_LANGUAGES: dict[str, str] = {
    "ar-eg": "ar-EG",
    "ar-sa": "auto",
    "ar-ae": "auto",
    "ar-qa": "auto",
    "ar-kw": "auto",
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


def build_tts(bundle: RuntimeBundle, settings: Settings) -> xai.TTS:
    """Construct the direct xAI Grok TTS with the bundle's pinned voice (leo/eve)."""
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
