"""Replay ONE real call through many models and compare latency + reply quality.

Method — forced canonical history. Every model answers the SAME turn with the SAME
context: the real system prompt (built by the worker's own assembler, so it is
byte-identical to production) plus the conversation so far. The assistant turns in that
history are generated ONCE by the control model and then frozen, so the models do not
diverge into different conversations — turn 5 is the same question for all of them.
Free-running each model would produce prettier transcripts and a useless comparison.

Fairness details that change the numbers:
  * One warmup call per model before timing. Without it the first model measured wears
    the TCP+TLS setup cost and looks slow.
  * N runs per turn, median reported. A single sample would have recorded the 22.3s
    stall seen in production (sim-4a22dc2a13f4) as that model's baseline.
  * Reasoning is disabled per family and VERIFIED from the response, not assumed:
    OpenAI's accepted value differs by generation ("none" on gpt-5.2+, "minimal" on
    gpt-5) and a rejected value would silently leave reasoning ON — which is the
    difference between a 0.6s and a 15s turn. reasoning_tokens is asserted to be 0.

    python model_comparisons/run_comparison.py            # full run
    python model_comparisons/run_comparison.py --runs 1   # quick smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "sim"))

from personas import get_market  # noqa: E402

from bluelab_runtime_bundle import RuntimeBundle, RuntimeConfig  # noqa: E402
from bluelab_voice.agent import build_system_prompt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# ── the models ──────────────────────────────────────────────────────────────────────
# `reasoning` is the value to send as OpenAI's reasoning_effort. None = not a reasoning
# model (or Claude). Probed at startup: the first value that returns reasoning_tokens==0
# wins, and a model where none works is dropped with a loud note rather than silently
# measured with reasoning left on.
CLAUDE = ["claude-sonnet-4-5", "claude-sonnet-5", "claude-haiku-4-5"]
# Deliberately small. gpt-5.2 and gpt-oss were requested; the other two are the picks
# that actually fit a voice turn — 5.4-mini is the current fast/cheap tier, and
# 5.3-chat-latest is on the NON-reasoning chat line, so there is no reasoning to
# disable and nothing that can silently re-enable it.
GPT = ["gpt-5.2", "gpt-5.4-mini", "gpt-5.3-chat-latest"]
# Not on the OpenAI account (checked: no `oss` model ids) — reachable only through
# LiveKit Inference, which is OpenAI-compatible, so it uses the same client with a
# different base_url + a short-lived LiveKit inference token.
INFERENCE_GPT = ["openai/gpt-oss-120b"]
CONTROL = "claude-sonnet-4-5"  # production model — generates the canonical history

# ── the conversation (real call sim-4a22dc2a13f4, 2026-08-01 19:41-19:43) ───────────
# Fragments split by the endpointing bug are merged back into the logical turns the
# rep actually spoke. TURN 5 is the one that triggered the 22.3s stall in production.
TURNS = [
    "ألو مساء الخير",
    "أنا يوسف بكلمك من شركة أليانز، ازيك عامل إيه",
    "كنت بكلمك بخصوص التأمين الصحي على الشركات، أنا من سيلز ميديكال انشورنس عندنا في "
    "شركة، فكنت بكلمك بشوف لو انت interested في حاجة زي كده",
    "اعرف تفاصيل أكتر عن الشركة، ممكن تقولي لي مثلا عدد الموظفين اللي عندكم",
    "لا لا انا بقول بس المعادي بس تلاتة، انما القاهرة كلها احنا عندنا يعني اكبر "
    "مستشفيات في القاهرة طبعا زي مثلا السعودي الالماني، عندك برضو مستشفى الجوي، كل دول معانا",
    "كنت بتقولي حاجة وقطعت",
]


@dataclass
class Result:
    model: str
    turn: int
    ttft: float
    total: float
    text: str
    out_tokens: int = 0
    reasoning_tokens: int = 0
    error: str = ""


@dataclass
class ModelSpec:
    name: str
    family: str  # "claude" | "gpt"
    reasoning: str | None = None
    skip: str = ""
    results: list[Result] = field(default_factory=list)


def system_prompt() -> str:
    """The REAL production prompt — same assembler the worker calls (ar-EG market)."""
    market = get_market("ar-EG")
    bundle = RuntimeBundle(
        attempt_id="cmp",
        org_id="cmp",
        livekit_room="attempt_cmp",
        call_type="discovery",
        lead_type="cold_outreach",
        language=market.code,
        wrapper_type="training",
        persona=market.persona,
        voice=market.voice,
        persona_gender=market.persona_gender,
        caller_gender=market.caller_gender,
        runtime_config=RuntimeConfig(),
        prompt_versions={},
        model_versions={},
    )
    return build_system_prompt(bundle)


# ── clients ─────────────────────────────────────────────────────────────────────────
def _anthropic():
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _openai():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _livekit_inference():
    """OpenAI-compatible client against LiveKit Inference (for models not on the OpenAI key)."""
    from livekit.agents.inference._utils import create_access_token, get_default_inference_url
    from openai import AsyncOpenAI

    token = create_access_token(
        os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"], ttl=3600
    )
    return AsyncOpenAI(api_key=token, base_url=get_default_inference_url())


async def probe_reasoning_off(client, model: str) -> tuple[str | None, str]:
    """Find the reasoning_effort value that yields reasoning_tokens == 0."""
    for value in ("none", "minimal", "low"):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "say OK"}],
                reasoning_effort=value,
                max_completion_tokens=64,
            )
            rt = (r.usage.completion_tokens_details.reasoning_tokens or 0) if r.usage else 0
            if rt == 0:
                return value, ""
        except Exception as exc:  # noqa: BLE001 — probing; a rejection is information
            msg = str(exc)
            if "max_tokens" in msg or "output limit" in msg:
                return value, ""  # it answered; the cap is the probe's, not a failure
            if "reasoning_effort" not in msg:
                return None, msg[:120]
    # Not a reasoning model at all → plain call works.
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "say OK"}],
            max_completion_tokens=64,
        )
        return None, ""
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "max_tokens" in msg or "output limit" in msg:
            return None, ""
        return None, msg[:120]


async def call_claude(client, model: str, system: str, msgs: list[dict]) -> Result:
    started = time.monotonic()
    ttft = None
    chunks: list[str] = []
    kwargs: dict = {}
    # Sonnet 5 thinks by default; every other Claude here does not. Voice needs it off.
    if model == "claude-sonnet-5":
        kwargs["thinking"] = {"type": "disabled"}
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=msgs,
        **kwargs,
    ) as stream:
        async for text in stream.text_stream:
            if ttft is None:
                ttft = time.monotonic() - started
            chunks.append(text)
        final = await stream.get_final_message()
    return Result(
        model=model,
        turn=-1,
        ttft=ttft or 0.0,
        total=time.monotonic() - started,
        text="".join(chunks),
        out_tokens=final.usage.output_tokens,
    )


async def call_gpt(client, model: str, reasoning: str | None, system: str, msgs: list[dict]) -> Result:
    started = time.monotonic()
    ttft = None
    chunks: list[str] = []
    kwargs: dict = {"reasoning_effort": reasoning} if reasoning else {}
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *msgs],
        max_completion_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    out_tokens = reasoning_tokens = 0
    async for ev in stream:
        if ev.usage:
            out_tokens = ev.usage.completion_tokens or 0
            details = ev.usage.completion_tokens_details
            reasoning_tokens = (details.reasoning_tokens or 0) if details else 0
        if ev.choices and ev.choices[0].delta.content:
            if ttft is None:
                ttft = time.monotonic() - started
            chunks.append(ev.choices[0].delta.content)
    return Result(
        model=model,
        turn=-1,
        ttft=ttft or 0.0,
        total=time.monotonic() - started,
        text="".join(chunks),
        out_tokens=out_tokens,
        reasoning_tokens=reasoning_tokens,
    )


async def one_call(spec: ModelSpec, clients: dict, system: str, msgs: list[dict]) -> Result:
    if spec.family == "claude":
        return await call_claude(clients["claude"], spec.name, system, msgs)
    return await call_gpt(clients[spec.family], spec.name, spec.reasoning, system, msgs)


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="timed runs per turn (median reported)")
    args = ap.parse_args()

    system = system_prompt()
    print(f"system prompt: {len(system)} chars")
    print(f"turns: {len(TURNS)}   runs/turn: {args.runs}\n")

    clients = {"claude": _anthropic(), "gpt": _openai(), "inference": _livekit_inference()}
    specs = [ModelSpec(m, "claude") for m in CLAUDE]

    print("probing reasoning-off value per GPT model…")
    for name, family in [(m, "gpt") for m in GPT] + [(m, "inference") for m in INFERENCE_GPT]:
        value, err = await probe_reasoning_off(clients[family], name)
        spec = ModelSpec(name, family, reasoning=value, skip=err)
        print(f"  {name:22} reasoning_effort={value or '(n/a)':8} {('SKIP: ' + err) if err else 'ok'}")
        specs.append(spec)
    specs = [s for s in specs if not s.skip]
    print()

    # 1. Canonical history from the control model — frozen, then reused by everyone.
    print(f"building canonical history with {CONTROL}…")
    control = next(s for s in specs if s.name == CONTROL)
    history: list[dict] = []
    canonical: list[str] = []
    for i, turn in enumerate(TURNS):
        history.append({"role": "user", "content": turn})
        res = await one_call(control, clients, system, history)
        canonical.append(res.text)
        history.append({"role": "assistant", "content": res.text})
        print(f"  turn {i + 1}: {len(res.text)} chars")
    print()

    # 2. Every model answers every turn against that same frozen history.
    for spec in specs:
        print(f"{spec.name}…", end=" ", flush=True)
        try:
            await one_call(spec, clients, system, [{"role": "user", "content": "ألو"}])  # warmup
        except Exception as exc:  # noqa: BLE001
            print(f"WARMUP FAILED: {str(exc)[:80]}")
            spec.skip = str(exc)[:120]
            continue
        for i, turn in enumerate(TURNS):
            ctx = [*history[: i * 2], {"role": "user", "content": turn}]
            runs: list[Result] = []
            for _ in range(args.runs):
                try:
                    runs.append(await one_call(spec, clients, system, ctx))
                except Exception as exc:  # noqa: BLE001
                    runs.append(Result(spec.name, i, 0, 0, "", error=str(exc)[:120]))
            ok = [r for r in runs if not r.error]
            if ok:
                best = Result(
                    model=spec.name,
                    turn=i,
                    ttft=statistics.median(r.ttft for r in ok),
                    total=statistics.median(r.total for r in ok),
                    text=ok[0].text,
                    out_tokens=ok[0].out_tokens,
                    reasoning_tokens=max(r.reasoning_tokens for r in ok),
                )
            else:
                best = runs[0]
                best.turn = i
            spec.results.append(best)
            print(".", end="", flush=True)
        print(" done")

    write_report(specs, system, canonical, args.runs)
    return 0


def write_report(specs: list[ModelSpec], system: str, canonical: list[str], runs: int) -> None:
    live = [s for s in specs if s.results and not s.skip]
    lines: list[str] = []
    add = lines.append

    add("# Model comparison — real call replay\n")
    add(f"Source call: **sim-4a22dc2a13f4** (2026-08-01 19:41–19:43), {len(TURNS)} turns.  ")
    add(f"System prompt: production assembler, ar-EG market, {len(system)} chars.  ")
    add(f"Method: forced canonical history (assistant turns frozen from `{CONTROL}`), ")
    add(f"{runs} timed runs per turn, **median** reported, one warmup per model.  ")
    add("Reasoning/thinking disabled on every model and verified from the response.\n")

    add("## Latency (median across turns)\n")
    add("| model | TTFT med | TTFT worst | total med | out tok med | reasoning tok |")
    add("|---|---|---|---|---|---|")
    for s in sorted(live, key=lambda s: statistics.median(r.ttft for r in s.results)):
        t = [r.ttft for r in s.results if not r.error]
        tot = [r.total for r in s.results if not r.error]
        ot = [r.out_tokens for r in s.results if not r.error]
        rt = max((r.reasoning_tokens for r in s.results), default=0)
        flag = " ⚠️" if rt else ""
        add(
            f"| `{s.name}` | **{statistics.median(t):.2f}s** | {max(t):.2f}s | "
            f"{statistics.median(tot):.2f}s | {int(statistics.median(ot))} | {rt}{flag} |"
        )
    add("")

    add("## Replies, turn by turn\n")
    add("Every model below answered the **same** question with the **same** history.\n")
    for i, turn in enumerate(TURNS):
        add(f"### Turn {i + 1}\n")
        add(f"**Rep says:** {turn}\n")
        add(f"**Production ({CONTROL}) said:** {canonical[i]}\n")
        add("| model | ttft | reply |")
        add("|---|---|---|")
        for s in sorted(live, key=lambda s: s.results[i].ttft if i < len(s.results) else 99):
            if i >= len(s.results):
                continue
            r = s.results[i]
            if r.error:
                add(f"| `{s.name}` | — | _error: {r.error}_ |")
            else:
                add(f"| `{s.name}` | {r.ttft:.2f}s | {r.text.replace(chr(10), ' ')} |")
        add("")

    skipped = [s for s in specs if s.skip]
    if skipped:
        add("## Skipped\n")
        for s in skipped:
            add(f"- `{s.name}` — {s.skip}")
        add("")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps(
            {
                s.name: [
                    {
                        "turn": r.turn,
                        "ttft": r.ttft,
                        "total": r.total,
                        "out_tokens": r.out_tokens,
                        "reasoning_tokens": r.reasoning_tokens,
                        "text": r.text,
                        "error": r.error,
                    }
                    for r in s.results
                ]
                for s in live
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'report.md'} and results.json")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
