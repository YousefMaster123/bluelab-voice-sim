"""Contract proof for the rebuilt agent<->backend boundary (no LiveKit import).

Verifies the parts that were REBUILT against the hardened backend, without needing the heavy
livekit-agents stack: the HMAC signing scheme matches the api's verifier byte-for-byte, the
callback payloads match the new request models, and a bundle emitted by the backend parses +
passes the knowledge-boundary guard. Runs in a lightweight venv (pydantic/httpx/tenacity/structlog).
"""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx

from bluelab_runtime_bundle import (
    PersonaSections,
    RuntimeBundle,
    RuntimeConfig,
    assert_runtime_safe,
)
from bluelab_voice.bundle_client import BundleClient
from bluelab_voice.callbacks import CallbackClient
from bluelab_voice.config import Settings
from bluelab_voice.signing import (
    SIGNATURE_HEADER,
    sign_body,
    verify_signature,
)

SECRET = "shared-agent-secret"


def _settings() -> Settings:
    return Settings(
        bluelab_agent_hmac_secret=SECRET,
        bluelab_api_url="http://api.test",
        callback_retries=1,
        bundle_fetch_retries=1,
    )


def _backend_verifier(secret: str, body: bytes) -> str:
    """Exact replica of the api's app/auth/agent.py `_compute_hmac` (the real gate)."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ── signing matches the backend's require_agent_hmac ──────────────────────────
def test_signature_matches_backend_verifier() -> None:
    body = b'{"attempt_id":"a1"}'
    headers = sign_body(SECRET, body)
    assert set(headers) == {SIGNATURE_HEADER}  # single header, NO timestamp
    assert headers[SIGNATURE_HEADER] == _backend_verifier(SECRET, body)
    assert verify_signature(SECRET, body, headers[SIGNATURE_HEADER])
    assert not verify_signature(SECRET, b"tampered", headers[SIGNATURE_HEADER])


def test_empty_body_signature_for_bundle_get() -> None:
    # The signed bundle GET signs over an empty body.
    assert sign_body(SECRET, b"")[SIGNATURE_HEADER] == _backend_verifier(SECRET, b"")


# ── transcript callback: payload shape + signature over the exact bytes ────────
async def test_transcript_payload_and_signature() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = req.content
        captured["sig"] = req.headers.get(SIGNATURE_HEADER)
        return httpx.Response(202, json={"status": "accepted", "inserted": 1, "received": 1})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api.test")
    cb = CallbackClient(_settings(), client=client)
    ok = await cb.send_transcript_segment(
        attempt_id="a1",
        sequence_index=0,
        speaker="participant",
        text="hi",
        timestamp_seconds=1.5,
        confidence=0.9,
    )
    await cb.aclose()

    assert ok
    assert captured["path"] == "/v1/callbacks/agent/transcript"
    payload = json.loads(captured["body"])  # type: ignore[arg-type]
    assert payload == {
        "attempt_id": "a1",
        "speaker": "participant",
        "text": "hi",
        "timestamp_seconds": 1.5,
        "sequence_index": 0,
        "confidence": 0.9,
    }
    assert "is_final" not in payload  # dropped for the hardened contract
    assert captured["sig"] == _backend_verifier(SECRET, captured["body"])  # type: ignore[arg-type]


# ── attempt-complete callback: only the contract fields go on the wire ─────────
async def test_attempt_complete_payload_is_trimmed() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = req.content
        return httpx.Response(202, json={"status": "accepted"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api.test")
    cb = CallbackClient(_settings(), client=client)
    ok = await cb.send_attempt_complete(
        attempt_id="a1",
        status="completed",
        duration_seconds=12.3,
        segment_count=5,  # worker bookkeeping — must NOT be sent
        end_of_call_report={"history": []},  # must NOT be sent
        error="none",  # must NOT be sent
    )
    await cb.aclose()

    assert ok
    payload = json.loads(captured["body"])  # type: ignore[arg-type]
    assert payload == {"attempt_id": "a1", "status": "completed", "duration_seconds": 12.3}
    for dropped in ("segment_count", "end_of_call_report", "error"):
        assert dropped not in payload


# ── bundle fetch: a backend-emitted bundle parses + validates ─────────────────
def _backend_bundle_dict() -> dict[str, object]:
    """Exactly what the hardened backend emits (RuntimeBundle.model_dump), incl. the corrected
    RuntimeConfig — using the SAME vendored schema the backend now mirrors."""
    return RuntimeBundle(
        attempt_id="a1",
        org_id="o1",
        livekit_room="attempt_a1",
        call_type="renewal",
        language="ar",
        wrapper_type="training",
        voice="eve",
        persona=PersonaSections(
            who_you_are="w",
            your_world="x",
            where_you_are_right_now="y",
            call_context="z",
        ),
        prompt_versions={"prospect": "p-1"},
        model_versions={"llm": "claude-haiku-4-5"},
    ).model_dump(mode="json")


async def test_bundle_fetch_parses_signs_and_validates() -> None:
    captured: dict[str, object] = {}
    body_dict = _backend_bundle_dict()

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["sig"] = req.headers.get(SIGNATURE_HEADER)
        return httpx.Response(200, json=body_dict)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api.test")
    bc = BundleClient(_settings(), client=client)
    bundle = await bc.fetch("a1")
    await bc.aclose()

    assert bundle.attempt_id == "a1"
    assert bundle.call_type == "renewal"
    # The corrected shared RuntimeConfig contract the agent's llm/stt/tts read.
    assert bundle.runtime_config.llm_model == "claude-haiku-4-5"
    assert bundle.runtime_config.stt_provider == "speechmatics"
    assert captured["path"] == "/v1/internal/attempts/a1/runtime-bundle"
    assert captured["sig"] == _backend_verifier(SECRET, b"")  # signed over the empty GET body
    assert assert_runtime_safe(bundle) is bundle


def test_runtime_config_carries_agent_read_fields() -> None:
    # The vendored RuntimeConfig must expose exactly the fields llm.py/stt.py/tts.py read AND that
    # the backend now emits — the fix that unblocked the agent<->backend bundle contract.
    rc = RuntimeConfig()
    for field in (
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "llm_max_tokens",
        "llm_prompt_caching",
        "stt_provider",
        "stt_language",
        "tts_provider",
    ):
        assert hasattr(rc, field), field
    assert rc.llm_model == "claude-haiku-4-5"
