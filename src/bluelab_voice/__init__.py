"""bluelab-voice — the latency-isolated LiveKit Agents worker hosting the Voice Prospect Agent.

See `worker.py` for the entrypoint. Per REPO-4 this package imports only the shared
`bluelab_runtime_bundle` contract + the provider SDKs (LiveKit Agents, Speechmatics, xAI,
Anthropic) + httpx for signed api calls — no Supabase client, no secrets beyond the scoped HMAC
and provider keys.
"""

from __future__ import annotations

__version__ = "0.1.0"
