"""Sim backend + 1-page frontend — drive the DEPLOYED voice agent with no real backend.

This stands in for `bluelab-backend-prod` on the only three jobs that matter for a live call, and
speaks the **identical contract**, so switching to the real backend later is one env var:

  1. mint a LiveKit join token that DISPATCHES the deployed agent  (what the frontend gets)
  2. serve  GET /v1/internal/attempts/{id}/runtime-bundle          (what the agent fetches)
  3. accept POST /v1/callbacks/agent/{transcript,attempt-complete} (what the agent reports)

What is REAL here: LiveKit Cloud, the deployed agent, Claude, Speechmatics, xAI, the actual
conversation, and the prompt (the preview calls the worker's own `build_system_prompt`, so what you
read on screen is byte-identical to what the model receives).
What is SIMULATED: the attempt row, persona generation, auth/JWT, scoring, and the database.

    python sim/server.py                       # http://localhost:8000  → open it in a browser

⚠ The DEPLOYED agent runs in LiveKit Cloud (Frankfurt) and fetches the bundle over the public
internet, so `localhost` is invisible to it. To drive the deployed agent you must expose this
server and point the agent at it, once:

    cloudflared tunnel --url http://localhost:8000
    lk agent update-secrets --secret BLUELAB_API_URL=https://<your-tunnel>

To drive a LOCAL worker instead (no tunnel needed), leave BLUELAB_API_URL=http://localhost:8000 in
.env and run `python -m bluelab_voice.worker dev` alongside this.

Pure stdlib http.server + the livekit api SDK (already a project dep). No FastAPI, no new deps.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parent.parent
_STATIC = Path(__file__).resolve().parent / "static"
# Make `src/` importable so the bundle is built from the REAL shared schema (guaranteed safe) and
# the prompt preview runs the REAL assembler.
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from personas import MARKETS, get_market  # noqa: E402

from bluelab_runtime_bundle import (  # noqa: E402
    RuntimeBundle,
    RuntimeConfig,
    assert_runtime_safe,
)

# Imported at MODULE level, on purpose: this pulls the LiveKit plugin stack, and LiveKit requires
# plugins to be registered on the MAIN thread. Importing it lazily inside a request handler (which
# runs on a worker thread of ThreadingHTTPServer) raises "Plugins must be registered on the main
# thread". Costs ~2s of startup; buys a prompt preview that is the worker's real assembler.
# GUARDRAILS_VERSION comes from the same module for a reason: the sim used to hardcode the version
# string, which silently drifted every time the guardrails were bumped (v21 bundle vs v22 code) and
# stamped the WRONG prompt version on every sim attempt record. Importing it makes drift impossible.
from bluelab_voice.agent import GUARDRAILS_VERSION, build_system_prompt  # noqa: E402

# PORT is injected by the host (Railway/Fly/Render set it); 8000 locally. HOST must stay 0.0.0.0 so
# the container is reachable from outside — binding 127.0.0.1 makes a deployed service unroutable.
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT") or 8000)


# ── env ─────────────────────────────────────────────────────────────────────────────────────────
def _load_env() -> None:
    """Load the repo `.env` into os.environ without overriding anything already set."""
    env_file = _REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# ── access guard ────────────────────────────────────────────────────────────────────────────────
# Optional shared key for the BROWSER-facing routes (`/`, `/sim/*`). Unset (the local default) =
# wide open. Set it whenever this is deployed to a public URL: `/sim/start` mints LiveKit tokens and
# summons the agent, so an open instance lets anyone burn your LiveKit / Anthropic / xAI credits.
#
# The agent-facing `/v1/*` routes are deliberately NOT guarded by this: the agent authenticates with
# the HMAC signature it already sends, and has no way to supply this key.
def _guarded(path: str) -> bool:
    return not path.startswith("/v1/")


def _authorized(supplied: str) -> bool:
    key = _env("SIM_ACCESS_TOKEN")
    if not key:
        return True
    return hmac.compare_digest(supplied or "", key)


# ── in-memory attempt store (the "database") ────────────────────────────────────────────────────
_LOCK = threading.Lock()
_ATTEMPTS: dict[str, dict] = {}


def _new_attempt(language: str) -> str:
    attempt_id = f"sim-{uuid.uuid4().hex[:12]}"
    with _LOCK:
        _ATTEMPTS[attempt_id] = {
            "language": language,
            "transcript": [],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
        }
    return attempt_id


def _attempt(attempt_id: str) -> dict | None:
    with _LOCK:
        return _ATTEMPTS.get(attempt_id)


# ── the bundle (identical shape to what the real api returns) ───────────────────────────────────
def build_bundle(attempt_id: str, language: str | None = None) -> RuntimeBundle:
    """Build + validate a runtime-safe bundle for an attempt, in the chosen market's language."""
    if language is None:
        rec = _attempt(attempt_id)
        language = (rec or {}).get("language")
    market = get_market(language)

    # Speechmatics: `ar_en` is the bilingual Arabic-English pack (all dialects + true intra-sentence
    # code-switching) — requires the DIRECT plugin path (SPEECHMATICS_API_KEY set), which sends the
    # code verbatim; LiveKit Inference would normalize it to ar-EN = Arabic-only. `en`/`fr` for the
    # non-Arabic markets.
    runtime_config = RuntimeConfig(
        # deepgram (nova-3) is the default: it beat Speechmatics ar_en on a real Egyptian clip,
        # which dropped ~a third of the words. Set SIM_STT_PROVIDER=speechmatics to switch back.
        # The worker maps ar_en -> "ar" for Deepgram, so stt_language stays as the market defines.
        stt_provider=_env("SIM_STT_PROVIDER", "deepgram"),
        stt_language=market.stt_language,
        stt_operating_point="enhanced",
        llm_model=_env("SIM_LLM_MODEL", "claude-sonnet-4-5"),
    )
    # TTS A/B switch: set SIM_TTS_VOICE_ID to a Fish reference id to run the whole sim on Fish
    # Audio instead of xAI. Env-driven on purpose — flipping providers for a listening test must
    # not need a redeploy of the agent. Unset → the xAI path, exactly as before.
    if fish_voice_id := _env("SIM_TTS_VOICE_ID"):
        runtime_config.tts_provider = "fishaudio"
        runtime_config.tts_voice_id = fish_voice_id
        runtime_config.tts_model = _env("SIM_TTS_MODEL", "s2.1-pro-free")
        runtime_config.tts_speed = float(_env("SIM_TTS_SPEED", "1.2"))
    bundle = RuntimeBundle(
        attempt_id=attempt_id,
        org_id="sim-org",
        livekit_room=f"attempt_{attempt_id}",
        call_type="discovery",
        lead_type="cold_outreach",
        language=market.code,  # ← the ONLY switch that selects the dialect block
        wrapper_type="training",
        persona=market.persona,
        voice=market.voice,
        persona_gender=market.persona_gender,
        caller_gender=market.caller_gender,
        runtime_config=runtime_config,
        prompt_versions={"voice_guardrails": GUARDRAILS_VERSION},
        model_versions={"voice_agent": runtime_config.llm_model},
    )
    return assert_runtime_safe(bundle)  # the same gate the worker applies


# ── prompt preview: runs the worker's OWN assembler, so it is byte-identical ─────────────────────
def _prompt_for(bundle: RuntimeBundle) -> str:
    """The exact prompt the model will receive for this bundle — no reconstruction."""
    return build_system_prompt(bundle)


# ── LiveKit token + agent dispatch (mirrors the real backend's _mint_livekit_token) ──────────────
def mint_token(room: str, attempt_id: str, identity: str) -> dict:
    """Mint a join token that ALSO tells LiveKit to dispatch our named agent into the room."""
    from livekit import api as livekit_api

    api_key, api_secret = _env("LIVEKIT_API_KEY"), _env("LIVEKIT_API_SECRET")
    agent_name = _env("LIVEKIT_AGENT_NAME", "bluelab-prospect")
    if not (api_key and api_secret):
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set in .env")

    grant = livekit_api.VideoGrants(
        room_join=True, room=room, can_publish=True, can_subscribe=True
    )
    token = (
        livekit_api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name("Sales rep (sim)")
        .with_grants(grant)
        .with_room_config(
            livekit_api.RoomConfiguration(
                agents=[
                    livekit_api.RoomAgentDispatch(
                        agent_name=agent_name,
                        # The worker prefers job metadata over the room-name fallback.
                        metadata=json.dumps({"attempt_id": attempt_id}),
                    )
                ]
            )
        )
        .to_jwt()
    )
    return {"room": room, "token": token, "url": _env("LIVEKIT_URL"), "agent_name": agent_name}


# ── HTTP ────────────────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence stdlib's noisy per-request logging
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict | list) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send(204, b"")

    def _check_access(self, path: str, query: dict) -> bool:
        """Enforce SIM_ACCESS_TOKEN on browser-facing routes; 401 + False when it fails."""
        if not _guarded(path):
            return True
        supplied = (query.get("k") or [""])[0] or self.headers.get("X-Sim-Key", "")
        if _authorized(supplied):
            return True
        self._json(401, {"error": "unauthorized — append ?k=<SIM_ACCESS_TOKEN> to the URL"})
        return False

    # ── GET ─────────────────────────────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if not self._check_access(path, query):
            return

        if path in ("/", "/index.html"):
            html = (_STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return

        if path == "/sim/markets":
            self._json(200, [{"code": m.code, "label": m.label} for m in MARKETS])
            return

        if path == "/sim/prompt-preview":
            language = (query.get("language") or [None])[0]
            try:
                bundle = build_bundle("preview", language=language)
                prompt = _prompt_for(bundle)
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(
                200,
                {
                    "language": bundle.language,
                    "voice": bundle.voice,
                    "stt_language": bundle.runtime_config.stt_language,
                    "llm_model": bundle.runtime_config.llm_model,
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                },
            )
            return

        if path == "/sim/events":
            attempt_id = (query.get("attempt_id") or [""])[0]
            after = int((query.get("after") or ["0"])[0])
            rec = _attempt(attempt_id)
            if rec is None:
                self._json(404, {"error": "unknown attempt"})
                return
            with _LOCK:
                lines = rec["transcript"][after:]
                status = rec["status"]
            self._json(200, {"status": status, "lines": lines, "next": after + len(lines)})
            return

        # ── the agent-facing contract ───────────────────────────────────────────────────────────
        if path.startswith("/v1/internal/attempts/") and path.endswith("/runtime-bundle"):
            attempt_id = path[len("/v1/internal/attempts/") : -len("/runtime-bundle")]
            try:
                bundle = build_bundle(attempt_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[sim] ✗ bundle build FAILED for {attempt_id!r}: {exc}", flush=True)
                self._json(500, {"error": str(exc)})
                return
            print(
                f"[sim] → served bundle  attempt={attempt_id}  language={bundle.language}  "
                f"voice={bundle.voice}  stt={bundle.runtime_config.stt_language}",
                flush=True,
            )
            self._send(200, bundle.model_dump_json().encode("utf-8"))
            return

        self._json(404, {"error": "not found", "path": path})

    # ── POST ────────────────────────────────────────────────────────────────────────────────────
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not self._check_access(path, parse_qs(parsed.query)):
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/sim/start":
            language = get_market(payload.get("language")).code
            attempt_id = _new_attempt(language)
            room = f"attempt_{attempt_id}"
            try:
                join = mint_token(room, attempt_id, identity=f"rep-{uuid.uuid4().hex[:8]}")
            except Exception as exc:  # noqa: BLE001
                print(f"[sim] ✗ token mint FAILED: {exc}", flush=True)
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            print(
                f"\n[sim] ▶ START  attempt={attempt_id}  language={language}  room={room}\n"
                f"[sim]   dispatching agent '{join['agent_name']}' via room config…",
                flush=True,
            )
            self._json(200, {"attempt_id": attempt_id, "language": language, **join})
            return

        # ── the agent-facing contract ───────────────────────────────────────────────────────────
        if path == "/v1/callbacks/agent/transcript":
            attempt_id = str(payload.get("attempt_id", ""))
            speaker = str(payload.get("speaker", "?"))
            text = str(payload.get("text", ""))
            seq = payload.get("sequence_index", "?")
            rec = _attempt(attempt_id)
            if rec is not None:
                with _LOCK:
                    rec["transcript"].append({"speaker": speaker, "text": text, "seq": seq})
            print(f"[sim] {seq:>3}  {speaker:>11}: {text}", flush=True)
            self._json(200, {"ok": True})
            return

        if path == "/v1/callbacks/agent/attempt-complete":
            attempt_id = str(payload.get("attempt_id", ""))
            status = str(payload.get("status", "?"))
            rec = _attempt(attempt_id)
            if rec is not None:
                with _LOCK:
                    rec["status"] = status
            print(
                f"[sim] ✔ COMPLETE  attempt={attempt_id}  status={status}  "
                f"duration={payload.get('duration_seconds')}s\n",
                flush=True,
            )
            self._json(200, {"ok": True})
            return

        self._json(404, {"error": "not found", "path": path})


def main() -> None:
    _load_env()
    # Fail fast if the schema/persona wiring is broken, before we start serving.
    for market in MARKETS:
        build_bundle("startup-check", language=market.code)

    api_url = _env("BLUELAB_API_URL", "http://localhost:8000")
    lk_url = _env("LIVEKIT_URL")
    agent_name = _env("LIVEKIT_AGENT_NAME", "bluelab-prospect")

    print(f"[sim] BlueLab sim listening on http://localhost:{PORT}  → open it in a browser")
    print(f"[sim]   markets : {', '.join(m.code for m in MARKETS)}")
    print(f"[sim]   livekit : {lk_url or '(LIVEKIT_URL not set!)'}")
    print(f"[sim]   agent   : {agent_name}")
    print(f"[sim]   agent will fetch its bundle from BLUELAB_API_URL = {api_url}")
    if "localhost" in api_url or "127.0.0.1" in api_url:
        print("[sim]   ↳ that is LOCAL: works for `worker dev` on this machine, but the DEPLOYED")
        print("[sim]     agent cannot reach it. Tunnel this server and run:")
        print("[sim]     lk agent update-secrets --secret BLUELAB_API_URL=https://<tunnel>")
    print("[sim] Ctrl+C to stop.\n", flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sim] shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
