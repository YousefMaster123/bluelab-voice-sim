"""HMAC request signing for the agent -> api callback boundary (API-9 / BE-18, 09 §3).

REBUILT for the hardened backend. The api's ``require_agent_hmac`` (app/auth/agent.py)
verifies a single header:

  - ``X-Agent-Signature`` = hex HMAC-SHA256(secret, raw_request_body)

over the EXACT bytes of the request body (empty for the signed bundle GET). The compare is
constant-time and fails closed (an unconfigured secret closes the boundary, never opens it).
This module is the worker side: it produces the header for both callbacks and the bundle fetch.

Note vs the MVP: the old scheme signed ``f"{timestamp}.{body}"`` and sent an extra
``X-BlueLab-Timestamp`` header. The hardened api signs the raw body only — no timestamp — so this
module was reduced to match the real verifier (verified against app/auth/agent.py).
"""

from __future__ import annotations

import hashlib
import hmac

SIGNATURE_HEADER = "X-Agent-Signature"


def compute_signature(secret: str, raw_body: bytes) -> str:
    """Hex HMAC-SHA256 of the raw body under the shared agent secret (matches the api)."""
    return hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def sign_body(secret: str, raw_body: bytes) -> dict[str, str]:
    """Return the ``X-Agent-Signature`` header for a raw body (empty bytes for a GET)."""
    return {SIGNATURE_HEADER: compute_signature(secret, raw_body)}


def verify_signature(secret: str, raw_body: bytes, signature: str) -> bool:
    """Constant-time verify (for tests / symmetry; the api owns the real gate)."""
    return hmac.compare_digest(compute_signature(secret, raw_body), signature)
