"""HMAC-signed callbacks to the api: transcript segments + attempt completion (07 §7, API-9/BE-18).

Two callbacks (backend app/api/routes/callbacks.py):
  - ``POST /v1/callbacks/agent/transcript``       — finalized segments, idempotent by
    ``(attempt_id, sequence_index, speaker)`` (transcript_segments UNIQUE). The endpoint accepts a
    single segment object (what we send) or a ``{"segments": [...]}`` batch.
  - ``POST /v1/callbacks/agent/attempt-complete`` — completion metadata -> api runs the Evaluator.

Signing: a single ``X-Agent-Signature`` header = HMAC-SHA256 over the exact raw body (see
``signing.py``). The api verifies it constant-time and fails closed.

REBUILT payloads to match the hardened backend's request models (engine/schemas.py):
  - TranscriptSegmentIn = {attempt_id, speaker, text, timestamp_seconds, sequence_index, confidence?}
    (the MVP's ``is_final`` field does not exist on the new contract — dropped from the wire).
  - AttemptCompleteIn   = {attempt_id, duration_seconds?, status}
    (the MVP's segment_count / end_of_call_report / error fields are not on the new contract —
    kept in this client's method signature for the worker's own bookkeeping/logging, but NOT sent).

Resilience (07 §10): failed transcript pushes are buffered in memory and flushed at end-of-call;
every outbound call has an explicit timeout + bounded retry.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import tenacity

from .config import Settings
from .logging import get_logger, get_request_id
from .signing import sign_body

_log = get_logger("bluelab.voice.callbacks")

_TRANSCRIPT_PATH = "/v1/callbacks/agent/transcript"
_COMPLETE_PATH = "/v1/callbacks/agent/attempt-complete"


class CallbackClient:
    """Signed, retrying client for the two agent callbacks; buffers failed transcript segments."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_base,
            timeout=httpx.Timeout(settings.callback_timeout_s),
        )
        self._buffer: list[dict[str, Any]] = []

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── transcript ────────────────────────────────────────────────────────────────────────
    async def send_transcript_segment(
        self,
        *,
        attempt_id: str,
        sequence_index: int,
        speaker: str,
        text: str,
        timestamp_seconds: float,
        is_final: bool = True,  # accepted for caller symmetry; not part of the wire contract
        confidence: float | None = None,
    ) -> bool:
        """Push one finalized transcript segment. Returns True on accept; buffers on hard failure.

        Idempotency is server-side on ``(attempt_id, sequence_index, speaker)``, so a retried or
        duplicated push is safe. We only buffer when the retry budget is exhausted.
        """
        payload: dict[str, Any] = {
            "attempt_id": attempt_id,
            "speaker": speaker,
            "text": text,
            "timestamp_seconds": timestamp_seconds,
            "sequence_index": sequence_index,
        }
        if confidence is not None:
            payload["confidence"] = confidence

        try:
            await self._post(_TRANSCRIPT_PATH, payload)
            return True
        except Exception as exc:  # noqa: BLE001 — buffer-and-continue, never lose a segment
            _log.warning(
                "transcript_buffered",
                attempt_id=attempt_id,
                sequence_index=sequence_index,
                error=str(exc),
            )
            self._buffer.append(payload)
            return False

    async def flush_buffer(self, attempt_id: str) -> int:
        """Retry any buffered segments (called at end of call). Returns the count still failing."""
        if not self._buffer:
            return 0
        pending = self._buffer
        self._buffer = []
        still_failing: list[dict[str, Any]] = []
        for payload in pending:
            try:
                await self._post(_TRANSCRIPT_PATH, payload)
            except Exception as exc:  # noqa: BLE001
                _log.error(
                    "transcript_flush_failed",
                    attempt_id=attempt_id,
                    sequence_index=payload.get("sequence_index"),
                    error=str(exc),
                )
                still_failing.append(payload)
        self._buffer = still_failing
        return len(still_failing)

    # ── completion ────────────────────────────────────────────────────────────────────────
    async def send_attempt_complete(
        self,
        *,
        attempt_id: str,
        status: str = "completed",
        duration_seconds: float | None = None,
        segment_count: int | None = None,  # worker bookkeeping; not on the wire contract
        end_of_call_report: dict[str, Any] | None = None,  # not on the wire contract
        error: str | None = None,  # not on the wire contract
    ) -> bool:
        """Signal end-of-call -> api runs the Evaluator. Idempotent by attempt_id.

        Only the hardened contract fields (attempt_id, status, duration_seconds) are sent; the
        richer args are accepted so the worker can pass what it tracks, and are logged locally.
        """
        payload: dict[str, Any] = {"attempt_id": attempt_id, "status": status}
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds

        try:
            await self._post(_COMPLETE_PATH, payload)
            _log.info(
                "attempt_complete_sent",
                attempt_id=attempt_id,
                status=status,
                segment_count=segment_count,
                had_error=bool(error),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("attempt_complete_failed", attempt_id=attempt_id, error=str(exc))
            return False

    # ── internals ─────────────────────────────────────────────────────────────────────────
    async def _post(self, path: str, payload: dict[str, Any]) -> None:
        # Sign over the EXACT bytes we send (canonical separators) so the api's HMAC over the raw
        # body matches byte-for-byte.
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()

        async for attempt in tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._settings.callback_retries),
            wait=tenacity.wait_exponential_jitter(initial=0.25, max=5.0),
            retry=tenacity.retry_if_exception_type(_RetryableCallback),
            reraise=True,
        ):
            with attempt:
                await self._post_once(path, raw)

    async def _post_once(self, path: str, raw: bytes) -> None:
        headers = sign_body(self._settings.bluelab_agent_hmac_secret, raw)
        headers["Content-Type"] = "application/json"
        if rid := get_request_id():
            headers["X-Request-Id"] = rid

        try:
            resp = await self._client.post(path, content=raw, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _RetryableCallback(f"transport error: {exc}") from exc

        if resp.status_code < 300:
            return
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _RetryableCallback(f"retryable status {resp.status_code}")
        # 4xx other than 429 — a signing/shape bug; retrying won't help. Surface it.
        raise RuntimeError(f"callback rejected with status {resp.status_code}: {resp.text[:200]}")


class _RetryableCallback(Exception):
    """Internal marker for callback failures worth retrying (5xx / 429 / transport)."""
