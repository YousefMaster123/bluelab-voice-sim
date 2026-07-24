"""Fetch + validate the runtime-safe bundle from the api (07 §6 step 3-4, REPO-4, VR-4).

The worker fetches its bundle over short-lived, scoped, signed service auth (09 §3): no Supabase
client, no tenant credentials — only the HMAC secret. The fetched payload is parsed into the shared
``RuntimeBundle`` schema and run through ``assert_runtime_safe``, so even if the api ever regressed
and emitted a forbidden field, the worker refuses it (defense-in-depth on the knowledge boundary).

If the bundle cannot be fetched, parsed, or validated, we raise ``BundleUnavailableError``; the
caller (worker) treats that as do-not-start-the-roleplay (VR-4): the attempt stays pending/failed.

REBUILT vs the MVP: the signed GET now sends the hardened ``X-Agent-Signature`` header (HMAC over
the empty body) — no timestamp header — matching the api's ``require_agent_hmac``.
"""

from __future__ import annotations

import httpx
import tenacity
from pydantic import ValidationError

from bluelab_runtime_bundle import (
    KnowledgeBoundaryError,
    RuntimeBundle,
    assert_runtime_safe,
)

from .config import Settings
from .logging import get_logger, get_request_id
from .signing import sign_body

_log = get_logger("bluelab.voice.bundle")


class BundleUnavailableError(RuntimeError):
    """The runtime bundle could not be fetched/parsed/validated — roleplay must not start (VR-4)."""


class BundleClient:
    """Thin signed client for GET-ing the runtime bundle for an attempt.

    One short-lived ``httpx.AsyncClient`` per worker; every call carries an explicit timeout and a
    bounded retry budget. 4xx (other than 429) is fatal — a malformed/forbidden bundle won't fix
    itself on retry — while 5xx/timeouts/429 are retried with backoff.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_base,
            timeout=httpx.Timeout(settings.bundle_fetch_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, attempt_id: str) -> RuntimeBundle:
        """Fetch + validate the bundle for ``attempt_id``. Raises ``BundleUnavailableError``."""
        path = f"/v1/internal/attempts/{attempt_id}/runtime-bundle"

        retryer = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._settings.bundle_fetch_retries),
            wait=tenacity.wait_exponential_jitter(initial=0.25, max=4.0),
            retry=tenacity.retry_if_exception_type(_RetryableFetch),
            reraise=True,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    return await self._fetch_once(path, attempt_id)
        except _RetryableFetch as exc:
            raise BundleUnavailableError(str(exc)) from exc
        except BundleUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — any unexpected error is do-not-start (fail-closed)
            raise BundleUnavailableError(f"unexpected bundle-fetch error: {exc}") from exc

        raise BundleUnavailableError("bundle fetch did not return")  # pragma: no cover

    async def _fetch_once(self, path: str, attempt_id: str) -> RuntimeBundle:
        # Signed, no-body GET: sign over the empty body so the api applies the same HMAC gate it
        # uses for callbacks (X-Agent-Signature over the raw request body).
        headers = sign_body(self._settings.bluelab_agent_hmac_secret, b"")
        if rid := get_request_id():
            headers["X-Request-Id"] = rid

        try:
            resp = await self._client.get(path, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            _log.warning("bundle_fetch_transport_error", attempt_id=attempt_id, error=str(exc))
            raise _RetryableFetch(f"transport error: {exc}") from exc

        if resp.status_code == 200:
            return self._parse(resp, attempt_id)

        if resp.status_code == 429 or resp.status_code >= 500:
            _log.warning(
                "bundle_fetch_retryable_status", attempt_id=attempt_id, status=resp.status_code
            )
            raise _RetryableFetch(f"retryable status {resp.status_code}")

        _log.error("bundle_fetch_fatal_status", attempt_id=attempt_id, status=resp.status_code)
        raise BundleUnavailableError(f"fatal bundle-fetch status {resp.status_code}")

    def _parse(self, resp: httpx.Response, attempt_id: str) -> RuntimeBundle:
        try:
            bundle = RuntimeBundle.model_validate_json(resp.content)
        except ValidationError as exc:
            _log.error("bundle_parse_failed", attempt_id=attempt_id, error=str(exc))
            raise BundleUnavailableError(f"bundle failed schema validation: {exc}") from exc

        try:
            assert_runtime_safe(bundle)
        except KnowledgeBoundaryError as exc:
            _log.error(
                "bundle_knowledge_boundary_violation", attempt_id=attempt_id, paths=exc.paths
            )
            raise BundleUnavailableError(str(exc)) from exc

        _log.info(
            "bundle_fetched_ok",
            attempt_id=attempt_id,
            call_type=bundle.call_type,
            language=bundle.language,
            voice=bundle.voice,
        )
        return bundle


class _RetryableFetch(Exception):
    """Internal marker for fetch failures worth retrying (5xx / 429 / transport)."""
