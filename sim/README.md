# `sim/` — drive the voice agent with no real backend

A stand-in for `bluelab-backend-prod` that speaks the **identical agent-facing contract**, plus a
one-page frontend to start calls and read the exact prompt. Dev-only; never deployed.

```
sim/
├── server.py          # the fake backend (stdlib http.server, no new deps)
├── personas.py        # one generic prospect, adapted per dialect
└── static/index.html  # the 1-page UI
```

## Run

```bash
uv sync                      # once
python sim/server.py         # http://localhost:8000 — open it in a browser
```

Pick a market → read the prompt → **Start call**. Transcript appears in the page and in the
server's stdout.

## The two modes

**A · Local worker (no tunnel).** Keep `BLUELAB_API_URL=http://localhost:8000` in `.env` and run
the worker on this machine:

```bash
python -m bluelab_voice.worker dev
```

**B · The DEPLOYED agent.** The cloud agent runs in Frankfurt and fetches its bundle over the
public internet, so `localhost` is invisible to it. Expose this server and point the agent at it
**once**:

```bash
cloudflared tunnel --url http://localhost:8000
lk agent update-secrets --secret BLUELAB_API_URL=https://<your-tunnel>
```

Switching back to the real backend later is the same one-line change — nothing in the agent
changes, because the contract is identical.

## Deployed (stable URL)

A quick Cloudflare tunnel gets a **new hostname every restart**, and you must re-point the agent
each time — so the sim is also deployable, for a URL you set once.

`railway.json` at the repo root points Railway's build at **`sim/Dockerfile`** (not the root
`Dockerfile`, which builds the agent worker). Note that config is repo-wide: if you ever deploy
*this repo* to Railway for something other than the sim, override the Dockerfile path on that
service.

Environment variables the deployed sim needs:

| Var | Why |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | mint join tokens + dispatch the agent |
| `LIVEKIT_AGENT_NAME` | defaults to `bluelab-prospect` |
| `SIM_ACCESS_TOKEN` | **set this when public** — see below |
| `SIM_LLM_MODEL` | optional, defaults to `claude-sonnet-4-5` |
| `PORT` | injected by the platform; do not set by hand |

`.env` is gitignored and never enters the image — set these in the platform's own env/secrets UI.

### `SIM_ACCESS_TOKEN`

Unset (local default) the sim is wide open. **Set it on any public deploy:** `/sim/start` mints
LiveKit tokens and summons the agent, so an open instance lets anyone burn your LiveKit / Anthropic
/ xAI credits. With it set, browser routes (`/`, `/sim/*`) need `?k=<token>` — open the page as
`https://…/?k=<token>` and the page forwards it on every call.

The agent-facing `/v1/*` routes stay open by design: the agent authenticates with the HMAC
signature it already sends and has no way to supply this key.

## What it implements

| Route | Who calls it | Stands in for |
| --- | --- | --- |
| `POST /sim/start` | the page | `POST /v1/drills/{id}/attempts` — mints a LiveKit token whose `roomConfig` dispatches `bluelab-prospect`, room `attempt_<id>` |
| `GET /v1/internal/attempts/{id}/runtime-bundle` | **the agent** | the real bundle endpoint (same schema, same `assert_runtime_safe` guard) |
| `POST /v1/callbacks/agent/transcript` | **the agent** | transcript sink |
| `POST /v1/callbacks/agent/attempt-complete` | **the agent** | completion sink |
| `GET /sim/prompt-preview?language=` | the page | — runs the worker's real `build_system_prompt` |
| `GET /sim/events?attempt_id=` | the page | — transcript polling |

HMAC signatures are **not verified** (the agent still signs them). Dev stand-in, not the hardened api.

## Real vs simulated

**Real:** LiveKit Cloud, the deployed agent, Claude, Speechmatics, xAI, the conversation, and the
prompt — the preview calls the worker's own assembler, so it is byte-identical to what the model gets.

**Simulated:** the attempt row, persona generation, auth/JWT, scoring, the database.

## Markets

`ar-QA` · `ar-AE` · `ar-SA` · `ar-KW` · `ar-EG` · `en` · `fr`

The dropdown sets the bundle's `language`, which is the **only** switch that selects the dialect
block in `bluelab_voice.agent._DIALECT_BLOCKS`. The persona is the same person in every market (a
~38-year-old HR manager facing a group-medical renewal) so that **dialect is the only variable**.

Override the model with `SIM_LLM_MODEL` (default `claude-sonnet-4-5`).
