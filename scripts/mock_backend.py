"""Local mock of the BlueLab api — lets you run the voice worker ALONE (no backend repo).

It stands in for the one HTTP dependency the worker can't start without: the runtime bundle.
It serves a valid, knowledge-boundary-safe ``RuntimeBundle`` for any attempt id and returns 200
for the two signed callbacks (transcript + attempt-complete), logging what the worker sends so you
can watch the call in this terminal.

It deliberately does NOT verify the ``X-Agent-Signature`` HMAC — the worker still signs every
request, we just don't check it. This is a dev stand-in, not the real hardened api.

Run (from the repo root, with the project installed so ``bluelab_runtime_bundle`` imports):

    python scripts/mock_backend.py            # listens on http://localhost:8000

Then in another terminal:

    python -m bluelab_voice.worker console

No third-party deps — pure stdlib ``http.server``. Only needs ``pydantic`` (already a project dep)
to build + validate the bundle against the real schema.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# Make ``src/`` importable so we build the bundle from the REAL shared schema (guaranteed safe).
sys.path.insert(0, str(_REPO / "src"))

from bluelab_runtime_bundle import (  # noqa: E402
    PersonaSections,
    RuntimeBundle,
    RuntimeConfig,
    assert_runtime_safe,
)

HOST, PORT = "127.0.0.1", 8000


def _stt_model() -> str:
    """Read STT_MODEL the same way the worker does (env, else the repo .env), default Speechmatics."""
    if val := os.environ.get("STT_MODEL"):
        return val.strip()
    env_file = _REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("STT_MODEL=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return "speechmatics/enhanced"


def _stt_profile() -> tuple[RuntimeConfig, str]:
    """Derive the STT runtime config + bundle language to MATCH the selected STT_MODEL.

    Deepgram nova-3 needs language ``multi``; Speechmatics uses the global ``ar`` model (which
    handles Egyptian-Arabic↔English code-switching). Both keep is_arabic_or_mixed True so the
    Egyptian-Arabic prompt block + ar-EG TTS stay on. This is what makes swapping STT a one-line
    change in .env.
    """
    model = _stt_model()
    if model.startswith("deepgram"):
        return (
            RuntimeConfig(stt_provider="deepgram", stt_language="multi", stt_operating_point="nova-3"),
            "multi",
        )
    # NOTE: `ar_en` is Speechmatics' real Arabic-English bilingual/code-switching pack. Testing
    # whether it survives LiveKit Inference (the SDK normalizes it to "ar-EN" before sending — it
    # may be rejected like `multi` was). If it errors, the reliable route is the direct Speechmatics
    # plugin; fall back to stt_language="ar" for the plain global Arabic model.
    return (
        RuntimeConfig(stt_provider="speechmatics", stt_language="ar_en", stt_operating_point="enhanced"),
        "ar",
    )


# ── A sample Egyptian-Arabic prospect (persona sections are used verbatim by the agent) ──────────
# Written in Egyptian colloquial Arabic (عامية مصرية). Third-person description of the person the
# agent embodies. Here: an HR manager the rep is trying to sell GROUP INSURANCE to. Same lean method
# — 2A is who she is, 2B is her factual world (insurance-relevant facts a rep would probe), 3 is the
# scene. No hidden-motive/trigger scripting (Section 1's engine handles reactivity). No product
# facts / rubric / answer key — the worker's guard enforces that boundary.
_PERSONA = PersonaSections(
    who_you_are=(
        "انتي مريم، عندك تمانية وتلاتين سنة، وانتي مديرة الـ إتش آر في شركة 'بيكسل بوينت' — شركة ديجيتال ماركتينج "
        "في المعادي. انتي إنسانة ودودة، وشاطرة في شغلك وعارفة ناسك وأرقامك كويس. انتي مش ساذجة — "
        "بتحبي تعرفي مين بيكلمك وهو عايز إيه قبل ما تدّي تفاصيل عن الشركة أو الأرقام. بس مش لازم "
        "تعملي استجواب: لو حسيتي إن الشخص جدّي، تقدري تكملي الكلام معاه وتسألي عن التفاصيل في سياق طبيعي."
    ),
    your_world=(
        "شركة 'بيكسل بوينت' فيها حوالي مية وتمانين موظف، أغلبهم شباب. عندكم دلوقتي تأمين طبي جماعي مع شركة "
        "تأمين بقالكم معاها كام سنة، بس الموظفين بقوا بيشتكوا منه على طول: صرف المطالبات بياخد وقت "
        "طويل، وشبكة المستشفيات والدكاترة محدودة، وناس كتير مش مبسوطة. وقرب معاد تجديد البوليصة، "
        "والشركة الحالية عايزة في التجديد سعر أعلى من السنة اللي فاتت. الإدارة ضاغطة عليكي تخلّي "
        "الموظفين مبسوطين وتظبطي التكلفة في نفس الوقت، وأي قرار بتأمين جديد لازم يعدّي على المدير "
        "المالي والإدارة. وانتي عارفة أرقامك كويس: عدد الموظفين، اللي بيغطيه التأمين الحالي، "
        "والميزانية اللي في إيدك."
    ),
    where_you_are_right_now=(
        "دلوقتي الصبح وانتي على مكتبك والتليفون رنّ من نمرة متعرفهاش، ورديتي. واللي بيكلمك راجل."
    ),
    call_context="",  # merged into Section 3 above; optional so it drops out of the prompt
)


def build_bundle(attempt_id: str) -> RuntimeBundle:
    """Build + validate a runtime-safe bundle for the given attempt id."""
    runtime_config, language = _stt_profile()  # derived from STT_MODEL so .env is the single switch
    # Testing Sonnet 4.6 for better Egyptian-Arabic dialect fidelity (Haiku drifts to فصحى on
    # business talk). Sonnet 4.6 still accepts the anthropic plugin's default sampling params, so
    # no plugin change is needed. Swap back to "claude-haiku-4-5" for the low-latency path.
    llm_model = "claude-sonnet-4-5"
    runtime_config = runtime_config.model_copy(update={"llm_model": llm_model})
    bundle = RuntimeBundle(
        attempt_id=attempt_id,
        org_id="local-dev-org",
        livekit_room=f"attempt_{attempt_id}",
        call_type="discovery",
        lead_type="cold_outreach",
        language=language,
        wrapper_type="training",
        persona=_PERSONA,
        voice="eve",  # female voice for Mona (HR manager)
        runtime_config=runtime_config,
        prompt_versions={"voice_guardrails": "voice-guardrails@v16"},
        model_versions={"voice_agent": llm_model},
    )
    return assert_runtime_safe(bundle)  # same gate the worker applies; fail loud if we regress


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence default noisy per-request logging
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path
        if path.startswith("/v1/internal/attempts/") and path.endswith("/runtime-bundle"):
            attempt_id = path[len("/v1/internal/attempts/") : -len("/runtime-bundle")]
            try:
                bundle = build_bundle(attempt_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[mock] bundle build FAILED for {attempt_id!r}: {exc}", flush=True)
                self._send(500, json.dumps({"error": str(exc)}).encode())
                return
            rc = bundle.runtime_config
            print(
                f"[mock] → served bundle for {attempt_id!r} "
                f"(STT {rc.stt_provider}/{rc.stt_language}, lang={bundle.language})",
                flush=True,
            )
            self._send(200, bundle.model_dump_json().encode())
            return
        self._send(404, json.dumps({"error": "not found", "path": path}).encode())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_unparsed": raw.decode("utf-8", "replace")}

        if self.path == "/v1/callbacks/agent/transcript":
            speaker = payload.get("speaker", "?")
            text = payload.get("text", "")
            seq = payload.get("sequence_index", "?")
            print(f"[mock] transcript #{seq:<3} {speaker:>11}: {text}", flush=True)
            self._send(200, b'{"ok":true}')
            return

        if self.path == "/v1/callbacks/agent/attempt-complete":
            print(
                f"[mock] ✔ attempt-complete: status={payload.get('status')} "
                f"duration={payload.get('duration_seconds')}s",
                flush=True,
            )
            self._send(200, b'{"ok":true}')
            return

        self._send(404, json.dumps({"error": "not found", "path": self.path}).encode())


def main() -> None:
    # Fail fast if the schema import/bundle is broken, before we start serving.
    probe = build_bundle("startup-check")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    prc = probe.runtime_config
    print(
        f"[mock] STT profile: model={_stt_model()} -> "
        f"{prc.stt_provider}/{prc.stt_language} (bundle language={probe.language})",
        flush=True,
    )
    print(f"[mock] BlueLab api mock listening on http://{HOST}:{PORT}", flush=True)
    print("[mock]   GET  /v1/internal/attempts/<id>/runtime-bundle", flush=True)
    print("[mock]   POST /v1/callbacks/agent/transcript", flush=True)
    print("[mock]   POST /v1/callbacks/agent/attempt-complete", flush=True)
    print("[mock] Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock] shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
