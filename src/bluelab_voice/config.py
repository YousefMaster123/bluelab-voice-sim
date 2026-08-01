"""Voice-agent settings (REPO-5: values from env only; REPO-4: no DB credentials here).

The worker holds exactly the secrets it needs to (a) join LiveKit rooms, (b) call Claude directly
for the LLM (Anthropic isn't in the LiveKit Inference catalog), and (c) fetch its bundle + sign
callbacks to the api. STT + TTS run through **LiveKit Inference** — billed via LiveKit Cloud, no
separate provider keys. It has **no** Supabase URL/key — the code-structure expression of the
knowledge boundary (REPO-4).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    # ── LLM: direct Anthropic (Claude isn't in the LiveKit Inference catalog) ──
    anthropic_api_key: str = ""

    # ── STT via LiveKit Inference (no key; speechmatics/enhanced, swappable via STT_MODEL). TTS
    # via the DIRECT xAI plugin (XAI_API_KEY) — Inference's xAI relay hit intermittent DNS
    # failures, and the direct path also unlocks the expressive speech tags. STT id verified
    # against the SDK's SpeechmaticsModels literal (`speechmatics/enhanced` | `speechmatics/standard`).
    stt_model: str = "speechmatics/enhanced"
    tts_model: str = "xai/tts-1"
    xai_api_key: str = ""
    # Direct Speechmatics plugin key. When set AND stt_model is speechmatics/*, the worker uses the
    # DIRECT plugin instead of LiveKit Inference — the only path that can send the raw bilingual
    # `ar_en` code (Inference normalizes ar_en → ar-EN = Arabic-only). Empty → Inference path.
    speechmatics_api_key: str = ""
    # Direct Fish Audio plugin key. Required when a bundle sets tts_provider="fishaudio" — Fish's
    # community/custom voices are NOT reachable through LiveKit Inference, only the direct plugin.
    fish_api_key: str = ""

    # ── LiveKit Cloud — worker registration + room join (07 §6) ──
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # ── BlueLab api — signed bundle fetch + HMAC callbacks (09 §3, REPO-4) ──
    # The ONLY api credential the worker holds is the scoped callback HMAC secret; there is no
    # Supabase client and no service-role key here.
    bluelab_api_url: str = "http://localhost:8000"
    bluelab_agent_hmac_secret: str = ""

    # Local standalone testing ONLY (never set in production): console mode carries no dispatch
    # metadata, so this pins an attempt_id for `python -m bluelab_voice.worker console`.
    # Field name keeps the bluelab_ prefix so the env var is BLUELAB_LOCAL_ATTEMPT_ID.
    bluelab_local_attempt_id: str = ""

    # External-call budgets (07 §10 — every outbound call has timeout + retry).
    bundle_fetch_timeout_s: float = 10.0
    bundle_fetch_retries: int = 3
    callback_timeout_s: float = 10.0
    callback_retries: int = 4
    # Replay-protection window the api enforces (09 §3: ±5 min). We sign with a fresh
    # timestamp per request; kept here for symmetry/testing.
    callback_timestamp_skew_s: int = 300

    @property
    def is_dev(self) -> bool:
        return self.environment.lower() in {"development", "dev", "local"}

    @property
    def api_base(self) -> str:
        return self.bluelab_api_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
