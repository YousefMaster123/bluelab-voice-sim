# bluelab-agent-prod — Voice Prospect Agent

Latency-isolated **LiveKit Agents** worker that plays the AI prospect in a live roleplay call.
It holds **no database client** (REPO-4): its only backend coupling is over HTTP — one signed
`GET` for the runtime bundle and two signed `POST` callbacks — all authenticated with a shared
HMAC secret.

## Layout
- `src/bluelab_voice/` — the worker: `worker.py` (entrypoint), `agent.py` (prompt assembly +
  session), `llm.py`/`stt.py`/`tts.py` (Claude / Speechmatics-Inference / xAI wiring),
  `bundle_client.py` (fetch runtime bundle), `callbacks.py` (transcript + attempt-complete),
  `signing.py` (HMAC), `transcript.py`, `config.py`, `logging.py`.
- `src/bluelab_runtime_bundle/` — **vendored** runtime-safe bundle contract + knowledge-boundary
  guard (canonical here; the backend mirrors the same schema).
- `tests/` — `test_contract.py` (agent↔backend boundary, no LiveKit), `test_guard.py` (bundle
  guard), `test_voice.py` (prompt/STT/TTS, needs the LiveKit stack). `scripts/smoke.py`.

## Backend contract (hardened)
- `GET /v1/internal/attempts/{id}/runtime-bundle` — signed, empty-body.
- `POST /v1/callbacks/agent/transcript` — `{attempt_id, speaker, text, timestamp_seconds, sequence_index, confidence?}`.
- `POST /v1/callbacks/agent/attempt-complete` — `{attempt_id, status, duration_seconds?}`.
- Auth: `X-Agent-Signature: hex(hmac_sha256(AGENT_HMAC_SECRET, raw_body))`. The agent's secret
  MUST equal the backend's `AGENT_HMAC_SECRET`.

## Run
```bash
python -m venv .venv && . .venv/Scripts/activate      # or bin/activate
pip install -e .                                       # installs livekit-agents + plugins
cp .env.example .env                                   # fill LiveKit + Anthropic + xAI + HMAC
python -m bluelab_voice.worker dev                     # register the worker with LiveKit Cloud
```

## Tests
```bash
pip install pydantic pydantic-settings httpx tenacity structlog pytest pytest-asyncio
PYTHONPATH=src pytest tests/test_contract.py tests/test_guard.py   # no LiveKit needed
PYTHONPATH=src pytest                                              # full suite (needs LiveKit + PyAV/ffmpeg)
```
