"""Run one audio file through an STT engine as if it were a LIVE CALL, and print what arrives.

The point is not just "what does it transcribe" — it is "what does the agent SEE, and when".
Audio is pushed in 20 ms frames at wall-clock speed through the streaming API, so interim and
final results land with the same timing they would during a real call. That exposes the two
things a batch transcript hides: how late finals arrive (transcription_delay in the worker logs)
and where an utterance gets split into multiple finals (the fragmented-transcript complaint).

    # Deepgram nova-3 via LiveKit Inference (no Deepgram account — uses LIVEKIT_API_KEY)
    python scripts/stt_probe.py "path/to/Recording.m4a" --model deepgram/nova-3 --language ar

    # Speechmatics, for comparison. NOTE: pass real-time (no --fast) — the direct plugin assumes
    # wall-clock input and returns almost nothing when fed as fast as possible.
    python scripts/stt_probe.py "path/to/Recording.m4a" --speechmatics --language ar_en

Do NOT use --language multi on Arabic: measured on a real Egyptian clip, nova-3 `multi`
mis-detected Spanish/Norwegian and returned nonsense.

Any container ffmpeg/libav can read works (m4a, mp3, wav, ogg); audio is resampled to the
engine's expected 16 kHz mono.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import av  # noqa: E402  (ships with livekit-agents)
import numpy as np  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import inference  # noqa: E402
from livekit.agents import stt as stt_api
from livekit.agents.utils import http_context  # noqa: E402

SAMPLE_RATE = 16_000
FRAME_MS = 20


def decode(path: Path) -> np.ndarray:
    """Decode any container to mono int16 at SAMPLE_RATE."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


async def run(engine: stt_api.STT, pcm: np.ndarray, realtime: bool) -> None:
    stream = engine.stream()
    started = time.monotonic()
    finals: list[str] = []

    async def feed() -> None:
        """Push 20 ms frames, sleeping between them so the engine sees a real-time stream."""
        step = SAMPLE_RATE * FRAME_MS // 1000
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step]
            stream.push_frame(
                rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
            )
            if realtime:
                await asyncio.sleep(FRAME_MS / 1000)
        stream.end_input()

    async def read() -> None:
        async for ev in stream:
            at = time.monotonic() - started
            if ev.type == stt_api.SpeechEventType.FINAL_TRANSCRIPT:
                text = ev.alternatives[0].text if ev.alternatives else ""
                if text.strip():
                    finals.append(text)
                    lang = ev.alternatives[0].language if ev.alternatives else "?"
                    print(f"[{at:6.2f}s] FINAL   ({lang})  {text}")
            elif ev.type == stt_api.SpeechEventType.INTERIM_TRANSCRIPT:
                text = ev.alternatives[0].text if ev.alternatives else ""
                if text.strip():
                    print(f"[{at:6.2f}s] interim        {text}")
            elif ev.type == stt_api.SpeechEventType.START_OF_SPEECH:
                print(f"[{at:6.2f}s] -- speech start --")
            elif ev.type == stt_api.SpeechEventType.END_OF_SPEECH:
                print(f"[{at:6.2f}s] -- speech end --")

    await asyncio.gather(feed(), read())
    await stream.aclose()

    print("\n" + "=" * 70)
    print(f"{len(finals)} final segment(s) — an utterance split across several is the")
    print("fragmentation that shows up as repeated transcript lines in the sim.\n")
    print(" ".join(finals))


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="audio file (m4a, wav, mp3 …)")
    ap.add_argument("--model", default="deepgram/nova-3", help="LiveKit Inference STT model id")
    ap.add_argument("--language", default="ar", help="language code for the engine")
    ap.add_argument(
        "--speechmatics",
        action="store_true",
        help="use the DIRECT Speechmatics plugin instead (the current production path)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="push audio as fast as possible instead of at wall-clock speed",
    )
    args = ap.parse_args()

    path = Path(args.audio)
    if not path.exists():
        print(f"no such file: {path}")
        return 1

    pcm = decode(path)
    secs = len(pcm) / SAMPLE_RATE
    label = "speechmatics (direct)" if args.speechmatics else args.model
    print(f"file    : {path.name}")
    print(f"audio   : {secs:.1f}s  {SAMPLE_RATE}Hz mono")
    print(f"engine  : {label}   language={args.language}")
    print(f"feed    : {'as fast as possible' if args.fast else 'real time (20ms frames)'}\n")

    async with http_context.open():
        if args.speechmatics:
            from speechmatics.voice import OperatingPoint

            from bluelab_voice.config import get_settings
            from bluelab_voice.stt import _RawLanguageSpeechmaticsSTT

            engine: stt_api.STT = _RawLanguageSpeechmaticsSTT(
                api_key=get_settings().speechmatics_api_key,
                raw_language=args.language,
                operating_point=OperatingPoint.ENHANCED,
            )
        else:
            engine = inference.STT(model=args.model, language=args.language)
        await run(engine, pcm, realtime=not args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
