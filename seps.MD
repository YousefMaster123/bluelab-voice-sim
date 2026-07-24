① USER clicks "Start call" in FRONTEND
        │  POST /v1/drills/{id}/attempts   (with the user's Supabase login token)
        ▼
② BACKEND does the setup:
     • checks drill published, rubric exists, consent given
     • creates a room name:  attempt_<uuid>
     • saves the attempt + an immutable snapshot (persona, product, prompts) in Postgres
     • mints a LiveKit token that ALSO says "dispatch agent 'bluelab-prospect' into this room"
     • returns  { attempt, livekit: {room, token, url} }  to the frontend
        │
        ▼
③ FRONTEND connects the user's MIC to the LiveKit room using that token
        │   (this is the ONLY thing the frontend does with LiveKit)
        ▼
④ LIVEKIT CLOUD sees the room has a dispatch request → summons the AGENT
        │
        ▼
⑤ AGENT wakes up in the room, sees room name "attempt_<uuid>", and now needs the persona.
     It does NOT receive it — it goes and FETCHES it:
        │  GET  {BLUELAB_API_URL}/v1/internal/attempts/<uuid>/runtime-bundle   (HMAC-signed)
        ▼
⑥ BACKEND returns the "runtime bundle" — persona + voice + language ONLY
     (deliberately NO scoring info; that's the "knowledge boundary")
        │
        ▼
⑦ AGENT starts talking:  ear=Speechmatics(STT) · brain=Claude(LLM) · mouth=xAI(TTS)
     As it goes, it pushes results BACK to the backend:
        POST /v1/callbacks/agent/transcript        (every line said)
        POST /v1/callbacks/agent/attempt-complete   (when the call ends)
        ▼
⑧ BACKEND stores the transcript, later scores it, writes the scorecard
        ▼
⑨ FRONTEND shows the live call, then the scorecard afterwards