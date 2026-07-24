"""Transcript streaming helper — assigns sequence indices + dispatches segment callbacks (07 §7).

The AgentSession event handlers (`user_input_transcribed`, `conversation_item_added`) are
synchronous and must not block, so this helper turns each finalized segment into a fire-and-forget
async task that POSTs to the api via `CallbackClient`. Idempotency is keyed on
`(attempt_id, sequence_index, speaker)` (DB-9), so we hand each segment a monotonically increasing
`sequence_index`. The agent's own STT is the only transcript source (VR-5).

`enqueue()` is non-blocking (schedules a task). `aclose()` awaits any in-flight dispatch tasks so
the shutdown hook can flush + send the completion callback only after every segment was attempted.
Buffering of segments that exhaust their retry budget is handled inside `CallbackClient` (never lose
a segment — 07 §10 / VR).
"""

from __future__ import annotations

import asyncio
import time

from .callbacks import CallbackClient
from .logging import get_logger

_log = get_logger("bluelab.voice.transcript")


class TranscriptStreamer:
    def __init__(self, *, attempt_id: str, callbacks: CallbackClient) -> None:
        self._attempt_id = attempt_id
        self._callbacks = callbacks
        self._seq = 0
        self._t0 = time.monotonic()
        self._tasks: set[asyncio.Task[bool]] = set()
        self.sent_count = 0

    def enqueue(self, *, speaker: str, text: str) -> None:
        """Schedule a finalized segment for delivery (non-blocking)."""
        seq = self._seq
        self._seq += 1
        ts = round(time.monotonic() - self._t0, 3)
        task = asyncio.create_task(self._deliver(seq, speaker, text, ts))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, seq: int, speaker: str, text: str, ts: float) -> bool:
        ok = await self._callbacks.send_transcript_segment(
            attempt_id=self._attempt_id,
            sequence_index=seq,
            speaker=speaker,
            text=text,
            timestamp_seconds=ts,
        )
        if ok:
            self.sent_count += 1
        return ok

    async def aclose(self) -> None:
        """Await all in-flight dispatch tasks (best-effort; failures are already buffered)."""
        if self._tasks:
            _log.info("transcript_draining", attempt_id=self._attempt_id, pending=len(self._tasks))
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
