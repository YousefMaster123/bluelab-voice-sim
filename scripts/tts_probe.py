"""Render one line of text through Fish Audio and/or xAI, side by side, to .wav files.

Purpose: audition a Fish Audio voice (including a community/custom voice from the playground,
which is reachable ONLY through the direct plugin — see below) on real Arabic BEFORE wiring
anything into the agent. Nothing here touches the worker, the bundle schema, or the prompt.

    # Fish only, using a voice's reference id from https://fish.audio/app/discovery/
    python scripts/tts_probe.py --fish-voice <reference_id> --text "ألو، مين معايا؟"

    # A/B against the current production voice (xAI eve)
    python scripts/tts_probe.py --fish-voice <reference_id> --xai-voice eve

Writes ./tts_probe_out/<provider>_<label>.wav and prints time-to-first-byte for each, so the
latency comparison is real rather than guessed.

WHY THE PLUGIN AND NOT LiveKit Inference: Inference exposes only Fish's own default voice
library. Per LiveKit's docs, "Pre-existing custom Fish Audio voices are not available through
LiveKit Inference" — community voices from the playground are custom voices, so auditioning one
requires a Fish account + FISH_API_KEY and the direct `livekit-plugins-fishaudio` plugin.

NOTE the plugin's TTS takes NO `language` argument (verified against the installed 1.6.7
signature): with Fish, the accent/dialect is a property of the chosen voice, not a per-call
knob. That is a real difference from the xAI path, which takes an `ar-EG`/`ar-SA`/`ar-AE` hint.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from livekit import rtc  # noqa: E402
from livekit.agents import tokenize  # noqa: E402
from livekit.agents.utils import http_context  # noqa: E402

# A neutral Egyptian line with a question, a name and a number — enough to hear pronunciation,
# prosody on the question mark, and how the voice handles digits.
DEFAULT_TEXT = "ألو، مساء الخير. أنا اسمي أسماء، بكلمك من شركة أليانز. عندك دقيقتين؟"


async def _render(engine: object, text: str, out: Path, label: str) -> None:
    """Synthesize `text` with `engine`, write a wav, and report ttfb + duration."""
    started = time.monotonic()
    ttfb: float | None = None
    frames: list[rtc.AudioFrame] = []
    stream = engine.synthesize(text)  # type: ignore[attr-defined]
    try:
        async for ev in stream:
            if ttfb is None:
                ttfb = time.monotonic() - started
            frames.append(ev.frame)
    finally:
        await stream.aclose()

    if not frames:
        print(f"  {label}: NO AUDIO returned")
        return

    combined = rtc.combine_audio_frames(frames)
    out.write_bytes(combined.to_wav_bytes())
    seconds = combined.samples_per_channel / combined.sample_rate
    print(
        f"  {label}: {out.name}  ttfb={ttfb:.2f}s  audio={seconds:.1f}s  "
        f"sample_rate={combined.sample_rate}"
    )


async def main() -> int:
    # Windows consoles default to cp1252, which cannot encode Arabic — same reconfigure the
    # worker's main() does, otherwise printing the probe text dies with UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=DEFAULT_TEXT, help="line to synthesize")
    ap.add_argument(
        "--fish-voice",
        help="Fish Audio reference id — the hex string in the playground URL fish.audio/m/<id>",
    )
    # `s2.1-pro-free` is the default because the paid models return HTTP 402 on an unfunded
    # account. It is not in the plugin's TTSModels literal, but `model` is typed `TTSModels | str`
    # so the string passes straight through to the API.
    ap.add_argument(
        "--fish-model", default="s2.1-pro-free", help="s2.1-pro-free | s2.1-pro | s2-pro | s1"
    )
    ap.add_argument("--fish-sample-rate", type=int, default=44100, help="match xAI's 44.1 kHz")
    ap.add_argument(
        "--fish-latency",
        default="normal",
        choices=["normal", "balanced", "low"],
        help="Fish streaming latency mode (plugin default is 'balanced'; production uses 'normal')",
    )
    # Fish `prosody.speed` multiplier. Repeatable so one run renders a ladder to pick from by ear;
    # أسماء reads ~50% longer than xAI eve on the same line at her natural 1.0.
    ap.add_argument(
        "--fish-speed",
        type=float,
        action="append",
        help="speed multiplier, repeatable (e.g. --fish-speed 1.2 --fish-speed 1.4)",
    )
    ap.add_argument("--xai-voice", help="also render with xAI for comparison (leo | eve)")
    ap.add_argument("--xai-language", default="ar-EG", help="xAI language hint")
    ap.add_argument("--out", default="tts_probe_out", help="output directory")
    args = ap.parse_args()

    if not args.fish_voice and not args.xai_voice:
        ap.error("give --fish-voice and/or --xai-voice")

    out_dir = _REPO / args.out
    out_dir.mkdir(exist_ok=True)
    print(f"text: {args.text}\nout : {out_dir}")

    if args.fish_voice:
        if not os.getenv("FISH_API_KEY"):
            print("  fish: FISH_API_KEY is not set — get one at https://fish.audio/go-api/")
        else:
            from livekit.plugins import fishaudio

            for speed in args.fish_speed or [None]:
                kwargs: dict[str, object] = {}
                if speed is not None:
                    kwargs["speed"] = speed
                fish = fishaudio.TTS(
                    model=args.fish_model,
                    voice_id=args.fish_voice,
                    sample_rate=args.fish_sample_rate,
                    # Match the production xAI setting: favour quality over shaving latency, so
                    # the comparison is quality-vs-quality, not one engine racing the other.
                    latency_mode=args.fish_latency,
                    **kwargs,  # type: ignore[arg-type]
                )
                suffix = "natural" if speed is None else f"{speed:g}x"
                await _render(
                    fish,
                    args.text,
                    out_dir / f"fish_{args.fish_voice[:8]}_{suffix}.wav",
                    f"fish {suffix}",
                )

    if args.xai_voice:
        if not os.getenv("XAI_API_KEY"):
            print("  xai: XAI_API_KEY is not set")
        else:
            from livekit.plugins import xai
            from livekit.plugins.xai import tts as _xai_tts

            # Production patches the plugin's hardcoded 24 kHz constant up to 44.1 kHz
            # (src/bluelab_voice/tts.py). Mirror it here or the A/B compares Fish against
            # something quieter/duller than what callers actually hear.
            _xai_tts.SAMPLE_RATE = 44100

            grok = xai.TTS(
                voice=args.xai_voice,
                language=args.xai_language,
                optimize_streaming_latency=0,
                # Same tokenizer fix the worker applies (see src/bluelab_voice/tts.py) — without
                # retain_format the streaming path drops word boundaries and pronunciation suffers.
                tokenizer=tokenize.basic.WordTokenizer(
                    ignore_punctuation=False, retain_format=True
                ),
            )
            await _render(grok, args.text, out_dir / f"xai_{args.xai_voice}.wav", "xai")

    return 0


async def _main_with_http() -> int:
    # The plugins fetch their aiohttp session from the worker's job context, which does not exist
    # in a plain script; http_context.open() provides one (the error message the plugins raise
    # without it recommends exactly this).
    async with http_context.open():
        return await main()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main_with_http()))
