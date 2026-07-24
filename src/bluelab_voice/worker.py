"""LiveKit Agents worker entrypoint — the Voice Prospect Agent worker (07 §6, AIR-6, REPO-4).

Dev command (spec): `python -m bluelab_voice.worker dev`
(or `uv run python -m bluelab_voice.worker dev`). `cli.run_app(server)` reads the
`dev` / `start` / `console` subcommand from argv.

Lifecycle (realizes 05 §12 / 07 §6):
  1. dispatched into the room by the backend → 2. read minimal metadata (attempt_id + IDs only —
     never payloads, 05 §6) → 3. fetch the runtime-safe bundle via signed service auth → 4. validate
     bundle shape + knowledge boundary (`assert_runtime_safe`, done inside BundleClient) →
     5. assemble the prompt (Section 1 + verbatim sections) → 6. connect STT→LLM→TTS →
     7. wait for participant audio → 8. open per call/lead type → 9. capture turns →
     10. stream transcript segments (idempotent) → 11. handle disconnect/reconnect → 12. clean
     shutdown, firing the attempt-complete callback. If the bundle can't be fetched/validated, the
     roleplay does NOT start (VR-4) — the worker reports a failed attempt and returns.

Isolation (REPO-4): imports only `packages/runtime-bundle` + the provider SDKs + httpx for the api
calls. No Supabase client, no secret beyond the scoped HMAC + provider keys.

All LiveKit SDK surfaces here (AgentServer, @server.rtc_session, JobContext, cli.run_app,
prewarm via setup_fnc, AgentSession events) were verified against docs.livekit.io at
implementation time (AIR-9).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from livekit.agents import (
    AgentFalseInterruptionEvent,
    AgentServer,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import ChatMessage

from .agent import ProspectAgent, build_session
from .bundle_client import BundleClient, BundleUnavailableError
from .callbacks import CallbackClient
from .config import get_settings
from .logging import configure_logging, get_logger, set_attempt_id
from .transcript import TranscriptStreamer

server = AgentServer()
_log = get_logger("bluelab.voice.worker")


def prewarm(proc: JobProcess) -> None:
    """Preload the Silero VAD once per process to absorb first-join latency (07 §6 prewarm)."""
    proc.userdata["vad"] = inference.VAD(model="silero")


# AgentServer setter form recommended by the docs over passing into the constructor.
server.setup_fnc = prewarm


def _attempt_id_from_metadata(ctx: JobContext) -> str | None:
    """Read attempt_id from job metadata (preferred) or room name (07 §6 step 2 — IDs only).

    Dispatch/room metadata carries only IDs, never payloads (05 §6 / AIR-2). We accept either a
    JSON job metadata blob with `attempt_id`, or a room named like `attempt_<id>` as a fallback.
    """
    raw = (ctx.job.metadata or "").strip()
    if raw:
        try:
            meta = json.loads(raw)
            if isinstance(meta, dict) and meta.get("attempt_id"):
                return str(meta["attempt_id"])
        except json.JSONDecodeError:
            _log.warning("metadata_not_json", room=ctx.room.name)
    room = ctx.room.name or ""
    if room.startswith("attempt_"):
        return room[len("attempt_") :]
    # Local standalone fallback (console mode carries no metadata/room name). Never set in prod.
    if local := get_settings().bluelab_local_attempt_id.strip():
        _log.warning("attempt_id_from_local_env", attempt_id=local)
        return local
    return None


@server.rtc_session(agent_name="bluelab-prospect")
async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    ctx.log_context_fields = {"room_name": ctx.room.name, "worker_id": ctx.worker_id}

    attempt_id = _attempt_id_from_metadata(ctx)
    if not attempt_id:
        # No usable identifier — cannot fetch a bundle, so cannot start the roleplay (VR-4).
        _log.error("no_attempt_id", room=ctx.room.name)
        return
    set_attempt_id(attempt_id)

    bundle_client = BundleClient(settings)
    callback_client = CallbackClient(settings)
    started = time.monotonic()

    try:
        # 3-4. Fetch + validate the runtime-safe bundle (knowledge boundary enforced inside).
        try:
            bundle = await bundle_client.fetch(attempt_id)
        except BundleUnavailableError as exc:
            _log.error("bundle_unavailable_no_start", attempt_id=attempt_id, error=str(exc))
            await callback_client.send_attempt_complete(
                attempt_id=attempt_id, status="failed", error=f"bundle_unavailable: {exc}"
            )
            return

        # 5-6. Assemble the prompt + wire STT→LLM→TTS with the bundle's runtime config.
        vad = ctx.proc.userdata.get("vad") or inference.VAD(model="silero")
        session = build_session(bundle, settings, vad=vad)
        streamer = TranscriptStreamer(attempt_id=attempt_id, callbacks=callback_client)

        # 9-10. Capture turns + stream transcript (the agent's own STT is the source, VR-5).
        @session.on("user_input_transcribed")
        def _on_user_transcribed(ev: UserInputTranscribedEvent) -> None:
            if ev.is_final and ev.transcript.strip():
                # STT PROOF (Deepgram nova-3 `multi`): `language` is the per-segment detected
                # language — watch it flip between ar/en as you code-switch mid-call (VR-1/07 §1).
                _log.info(
                    "stt_final",
                    attempt_id=attempt_id,
                    language=ev.language,
                    chars=len(ev.transcript),
                    text=ev.transcript,
                )
                streamer.enqueue(speaker="participant", text=ev.transcript)
            elif ev.transcript.strip():
                _log.debug("stt_interim", language=ev.language, text=ev.transcript)

        @session.on("conversation_item_added")
        def _on_item_added(ev: ConversationItemAddedEvent) -> None:
            item = ev.item
            if isinstance(item, ChatMessage) and item.role == "assistant" and item.text_content:
                # Enqueue first; the agent (the AI prospect) is the "prospect" speaker (DB enum).
                streamer.enqueue(speaker="prospect", text=item.text_content)
                # `interrupted` = this reply was cut off by a barge-in (INTERRUPTION PROOF). Log
                # length only for the text — it can contain non-ASCII that crashes the Windows
                # console encoder, and a logging crash here would abort the enqueue above.
                _log.info(
                    "agent_reply",
                    attempt_id=attempt_id,
                    chars=len(item.text_content),
                    interrupted=getattr(item, "interrupted", None),
                )

        # ── Diagnostic logging so a manual test can SEE each feature firing (local dev aid). ──
        # TURN DETECTION PROOF: the state machine listening → thinking → speaking. The
        # listening→thinking transition IS the multilingual end-of-turn detector deciding your turn
        # ended; speaking→listening is it handing the turn back.
        @session.on("agent_state_changed")
        def _on_agent_state(ev: AgentStateChangedEvent) -> None:
            _log.info("agent_state", attempt_id=attempt_id, frm=ev.old_state, to=ev.new_state)

        # VAD PROOF: user speaking/listening as detected by the Silero VAD driving turn-taking.
        @session.on("user_state_changed")
        def _on_user_state(ev: UserStateChangedEvent) -> None:
            _log.info("user_state", attempt_id=attempt_id, frm=ev.old_state, to=ev.new_state)

        # ADAPTIVE INTERRUPTION PROOF: emitted only when interruption mode="adaptive" is actually
        # running (not the VAD fallback). `is_interruption` False = it classified your overlap as a
        # backchannel and did NOT stop the agent; True = a real barge-in. Untyped: the event class
        # isn't exported by this SDK build, so we read fields defensively.
        @session.on("overlapping_speech")
        def _on_overlap(ev: Any) -> None:
            _log.info(
                "adaptive_interruption",
                attempt_id=attempt_id,
                is_interruption=getattr(ev, "is_interruption", None),
                probability=round(getattr(ev, "probability", 0.0) or 0.0, 3),
                detection_delay=round(getattr(ev, "detection_delay", 0.0) or 0.0, 3),
                num_requests=getattr(ev, "num_requests", None),
            )

        # Adaptive model caught a false interruption (backchannel) and resumed the agent's turn.
        @session.on("agent_false_interruption")
        def _on_false_interruption(ev: AgentFalseInterruptionEvent) -> None:
            _log.info(
                "false_interruption_filtered",
                attempt_id=attempt_id,
                resumed=getattr(ev, "resumed", None),
            )

        # PREEMPTIVE GENERATION PROOF (indirect): speech created via generate_reply before/at turn
        # end. Watch the timing vs. metric_llm.ttft below — preemptive gen starts the LLM before the
        # user's turn fully commits, so first-token latency stays low.
        @session.on("speech_created")
        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            _log.info(
                "speech_created",
                attempt_id=attempt_id,
                source=ev.source,
                user_initiated=ev.user_initiated,
            )

        # LATENCY / CACHING PROOF: per-turn metrics. EOU delay = turn-detector timing; LLM ttft +
        # cached_tokens = preemptive gen + prompt caching working; InterruptionMetrics counts real
        # interruptions vs. filtered backchannels for the adaptive model.
        @session.on("metrics_collected")
        def _on_metrics(ev: MetricsCollectedEvent) -> None:
            m = ev.metrics
            kind = type(m).__name__
            if kind == "EOUMetrics":
                _log.info(
                    "metric_turn_detection",
                    attempt_id=attempt_id,
                    eou_delay=round(getattr(m, "end_of_utterance_delay", 0.0) or 0.0, 3),
                    transcription_delay=round(getattr(m, "transcription_delay", 0.0) or 0.0, 3),
                )
            elif kind == "LLMMetrics":
                _log.info(
                    "metric_llm",
                    attempt_id=attempt_id,
                    ttft=round(getattr(m, "ttft", 0.0) or 0.0, 3),
                    cancelled=getattr(m, "cancelled", None),
                    cached_tokens=getattr(m, "prompt_cached_tokens", None),
                    prompt_tokens=getattr(m, "prompt_tokens", None),
                )
            elif kind == "InterruptionMetrics":
                _log.info(
                    "metric_adaptive_interruption",
                    attempt_id=attempt_id,
                    num_interruptions=getattr(m, "num_interruptions", None),
                    num_backchannels=getattr(m, "num_backchannels", None),
                    num_requests=getattr(m, "num_requests", None),
                    detection_delay=round(getattr(m, "detection_delay", 0.0) or 0.0, 3),
                )
            elif kind == "TTSMetrics":
                _log.info(
                    "metric_tts",
                    attempt_id=attempt_id,
                    ttfb=round(getattr(m, "ttfb", 0.0) or 0.0, 3),
                    streamed=getattr(m, "streamed", None),
                )

        # Graceful degradation (07 §10): recreate-per-response components may continue; STT resets.
        @session.on("error")
        def _on_error(ev: ErrorEvent) -> None:
            _log.error(
                "session_error",
                attempt_id=attempt_id,
                source=type(ev.source).__name__,
                recoverable=ev.error.recoverable,
                error=str(ev.error),
            )

        # 12. Clean shutdown — flush buffered transcript + fire attempt-complete (BE-16).
        async def _on_shutdown() -> None:
            await streamer.aclose()
            remaining = await callback_client.flush_buffer(attempt_id)
            report = _end_of_call_report(session)
            await callback_client.send_attempt_complete(
                attempt_id=attempt_id,
                status="completed",
                duration_seconds=round(time.monotonic() - started, 3),
                segment_count=streamer.sent_count,
                end_of_call_report=report,
                error=(f"{remaining} transcript segment(s) unflushed" if remaining else None),
            )
            await callback_client.aclose()
            await bundle_client.aclose()

        ctx.add_shutdown_callback(_on_shutdown)

        # 6/7. Start the session (connects to the room) and wait for the participant.
        # close_on_disconnect=False: React dev StrictMode double-mounts the call page, which
        # disconnects+reconnects the participant; without this the agent session would die on the
        # first (spurious) disconnect (matches the reference agent).
        await session.start(
            agent=ProspectAgent(bundle),
            room=ctx.room,
            room_options=room_io.RoomOptions(close_on_disconnect=False),
        )

        # 8. Open the call — you've just picked up. Answer like a real person answers a phone: a
        # brief hello, nothing more. Do NOT interrogate the caller or unload your mood before you
        # even know who it is — that reads as unnatural/aggressive for a call opening.
        await session.generate_reply(
            instructions="You've just picked up the phone in character, not knowing who's calling. "
            "Open exactly the way a real person answers a call — a short, natural hello and nothing "
            "more. Don't ask who they are yet, don't explain yourself, don't unload; just answer and "
            "let them talk."
        )

    except Exception as exc:  # noqa: BLE001 — last-resort guard so a crash still reports + cleans up
        _log.error("worker_unhandled", attempt_id=attempt_id, error=str(exc), exc_info=exc)
        await callback_client.send_attempt_complete(
            attempt_id=attempt_id, status="failed", error=f"worker_error: {exc}"
        )
        await callback_client.aclose()
        await bundle_client.aclose()
        raise


def _end_of_call_report(session: Any) -> dict[str, Any]:
    """Best-effort authoritative end-of-call summary (07 §7). Never raises."""
    try:
        history = session.history
        items = history.to_dict() if hasattr(history, "to_dict") else None
        return {"history": items} if items is not None else {}
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    # Windows consoles default to cp1252, which can't encode Arabic — reconfigure stdio to UTF-8 so
    # LiveKit's logging of Arabic transcripts/replies doesn't spew UnicodeEncodeError tracebacks.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    settings = get_settings()
    # The LiveKit Agents CLI reads LIVEKIT_URL / API_KEY / API_SECRET from os.environ; mirror them
    # from our pydantic settings (which load .env) so `python -m bluelab_voice.worker dev` works
    # without a separate `export` step (REPO-5: values still originate from .env, not code).
    if settings.livekit_url:
        os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
    if settings.livekit_api_key:
        os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
    if settings.livekit_api_secret:
        os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
    configure_logging(settings.log_level, json=not settings.is_dev)
    cli.run_app(server)


if __name__ == "__main__":
    main()
