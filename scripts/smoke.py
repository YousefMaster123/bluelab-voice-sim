"""In-process smoke check (no network / no LiveKit connection). Run: python scripts/smoke.py

Verifies the worker module imports (the full LiveKit SDK wiring), the prompt assembles
knowledge-bounded, the bundle guard rejects a forbidden payload, and HMAC signing round-trips under
the new X-Agent-Signature scheme. Requires the livekit deps installed.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import bluelab_voice.worker  # noqa: E402,F401  — import the full graph (LiveKit SDK wiring)
from bluelab_runtime_bundle import (  # noqa: E402
    PersonaSections,
    RuntimeBundle,
    assert_payload_safe,
)
from bluelab_voice.agent import SECTION_1_GUARDRAILS, build_system_prompt  # noqa: E402
from bluelab_voice.signing import SIGNATURE_HEADER, sign_body, verify_signature  # noqa: E402


def main() -> None:
    bundle = RuntimeBundle(
        attempt_id="att_x",
        org_id="org_x",
        livekit_room="attempt_att_x",
        call_type="renewal",
        language="mixed",
        wrapper_type="training",
        voice="eve",
        persona=PersonaSections(
            who_you_are="You are Salma, a long-time customer weighing renewal.",
            your_world="You have held the policy three years; mixed experience.",
            where_you_are_right_now="Open but price-sensitive.",
            call_context="Renewal call; you know the rep and your account history.",
        ),
    )
    prompt = build_system_prompt(bundle)
    assert prompt.startswith(SECTION_1_GUARDRAILS), "guardrails must lead (prompt caching)"
    assert "Salma" in prompt and "CALL CONTEXT" in prompt
    print("prompt assembly   ok  (", len(prompt), "chars )")

    try:
        assert_payload_safe({"rubric": {"dimensions": []}})
        raise AssertionError("guard should have rejected a rubric payload")
    except ValueError:
        print("knowledge guard   ok  (rubric payload rejected)")

    headers = sign_body("s3cret", b"{}")
    assert set(headers) == {SIGNATURE_HEADER}
    assert verify_signature("s3cret", b"{}", headers[SIGNATURE_HEADER])
    print("hmac signing      ok  (X-Agent-Signature)")

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
