"""bluelab-runtime-bundle — the runtime-safe bundle contract + knowledge-boundary guard.

Shared source of truth (REPO-3) imported by `apps/api` (which builds bundles) and
`apps/voice-agent` (which validates them, REPO-4). Public surface:

    from bluelab_runtime_bundle import (
        RuntimeBundle, PersonaSections, RuntimeConfig,
        assert_runtime_safe, assert_payload_safe, find_forbidden_fields,
        KnowledgeBoundaryError, FORBIDDEN_FIELDS,
    )
"""

from __future__ import annotations

from .guard import (
    FORBIDDEN_FIELDS,
    KnowledgeBoundaryError,
    assert_payload_safe,
    assert_runtime_safe,
    find_forbidden_fields,
)
from .schema import (
    CallType,
    LeadType,
    PersonaSections,
    RuntimeBundle,
    RuntimeConfig,
    Voice,
    WrapperType,
)

__all__ = [
    "FORBIDDEN_FIELDS",
    "CallType",
    "KnowledgeBoundaryError",
    "LeadType",
    "PersonaSections",
    "RuntimeBundle",
    "RuntimeConfig",
    "Voice",
    "WrapperType",
    "assert_payload_safe",
    "assert_runtime_safe",
    "find_forbidden_fields",
]

__version__ = "0.1.0"
