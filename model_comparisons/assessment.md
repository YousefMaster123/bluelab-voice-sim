# Quality assessment — Egyptian dialect, character, and errors

Hand-written analysis of the 42 replies in `report.md` (7 models × 6 turns). Latency and
token counts are measured; everything below is a language and character judgement, with
the evidence quoted so it can be argued with.

Scored on three axes:

- **Dialect** — is it Egyptian colloquial, or does it drift toward MSA / another dialect?
- **Character** — does it hold مريم's persona: a guarded HR manager who keeps asking how a
  cold caller got her number, and does not hand over information he hasn't earned?
- **Errors** — anything a listener would hear as wrong.

---

## Ranking

| rank | model | dialect | character | verdict |
|---|---|---|---|---|
| 1 | `gpt-5.3-chat-latest` | **best in set** | too forthcoming | best Arabic; character is prompt-tunable |
| 2 | `claude-haiku-4-5` | strong | **strong** | quality-neutral vs production at ⅓ the cost |
| 3 | `claude-sonnet-4-5` *(control)* | strong | **strong** | the incumbent; nothing wrong with it |
| 4 | `gpt-5.2` | strong | breaks early | leaks persona details unprompted |
| 5 | `claude-sonnet-5` | strong | over-cooperative | good Arabic, wrong shape for voice |
| — | `gpt-5.4-mini` | mixed | **fails** | ruled out — errors + premature hangup |
| — | `openai/gpt-oss-120b` | mixed | **fails** | ruled out — comprehension failures |

---

## `gpt-5.3-chat-latest` — best Egyptian in the set

**Dialect: excellent.** Produced native constructions no other model reached for.

> Turn 3 — `احنا عندنا تأمين طبي شغالين بيه بقالنا شوية، بس بصراحة فيه مشاكل والموضوع داخل على تجديد`

`بقالنا شوية` and `داخل على تجديد` are idiomatic Egyptian, not translated MSA. Its
code-switching is also the most natural: `أنا مش بطلع داتا عن الشركة` uses `داتا` the way an
Egyptian office worker actually does, rather than reaching for a formal Arabic equivalent.

**Character: good pushback, but too open.** Turn 5 is the strongest single reply in the run —

> `إنت كده بتبيعلي قبل ما حتى نعرف هنكمل مع بعض ولا لأ` … `رد عليّ في النقطة دي الأول وبعدين نشوف`

— it names the manipulation and refuses to move on. But at Turn 3 it volunteers that the
current insurance has problems and is up for renewal, which is leverage مريم shouldn't give
away that early.

**Errors:** Turn 2 `مين معايا تاني من أليانز` is slightly off — the rep had just introduced
himself. Minor.

**Verdict:** the dialect is the hardest thing to fix and this model has the best of it.
Being too forthcoming is a prompt problem, not a model problem. Worth a live A/B — it is
also **0.81s TTFT vs 1.32s** and **51 median tokens vs 77**, so it would shorten calls.

---

## `claude-haiku-4-5` — quality-neutral vs production, one third the cost

**Dialect: strong.** Natural throughout, with correct Egyptian negation (`ما...ش`) and
domain vocabulary that fits the role:

> Turn 3 — `شبكة الدكاترة وصرف المطالبات` — the two things an HR manager actually evaluates.

**Character: strong.** Holds the guarded line across every turn, always returning to how the
caller got her number, without becoming repetitive:

> Turn 5 — `بس أنا لسه ما عرفتش حضرتك وصلت لي منين؟ يعني في حد من الشركة كلمك، ولا حضرتك جبت النمرة إزاي؟`

**Errors:** one slip — Turn 6 `لا أنا ما قلت حاجة` should be `ما قلتش حاجة`. Audible to a
native speaker but minor. Emitted `[pause]` at Turn 2 (see the speech-tag note below).

**Verdict:** matches the incumbent on both axes. The case for switching is **cost (3×
cheaper), not latency** — the measured TTFT gain over `claude-sonnet-4-5` is only 0.16s,
which does not justify the change on its own.

---

## `claude-sonnet-4-5` — the incumbent, and it is fine

**Dialect: strong.** Clean, unforced Egyptian: `جبت نمرتي منين`, `ولا دي مكالمة عامة كده`.

**Character: strong.** The most consistent persona discipline in the run — it never lets the
rep move past the unanswered question, which is exactly what the persona is written to do.

**Errors:** one typo across six turns (`بالذبط` for `بالظبط`).

**Verdict:** nothing here is broken. Any switch should be justified by cost or reply length,
not by fixing a quality problem — there isn't one.

---

## `gpt-5.2` — good language, breaks character early

**Dialect: strong.** `مين اللي إدّاك رقمنا`, `عشان أبقى ماشية صح`, `كلامك اتغيّر شوية` are all
natural. Used `[sigh]` expressively at Turn 5.

**Character: fails at Turn 2.**

> `أنا مريم، مديرة الـHR في بيكسل بوينت، الحمدلله تمام`

She hands a cold caller her name, title and company in the first exchange. The persona is
guarded and this is the opposite; it also removes the discovery work the rep is supposed to
be practising.

**Verdict:** would need explicit prompt work to stop volunteering identity. Same class of
problem as `gpt-5.3-chat-latest` but worse, and its Arabic is not as good.

---

## `claude-sonnet-5` — good Arabic, wrong shape for a phone call

**Dialect: strong.** No complaints on the language itself.

**Character: over-cooperative, and long.** Turn 6 runs to a full paragraph that concedes the
renewal timing, offers to review a deck, and drops the unanswered question entirely:

> `احنا فعلاً عندنا بوليصة تأمين حالياً وقربت تتجدد، فممكن نتكلم … ابعتلي بريزنتيشن`

**Measured cost:** **111 median output tokens** — the most in the set — and **4.38s median
total generation**, the slowest. For voice that is 15+ seconds of speaking per turn.

**Verdict:** not the upgrade path. No TTFT gain, longest replies, and it gives ground the
persona should hold.

---

## `gpt-5.4-mini` — ruled out

Fastest model measured (0.78s TTFT), and the one that makes the most mistakes.

- **Turn 1:** rep says **مساء الخير** (good evening); model answers **صباح الخير** (good
  morning). An attention error on the simplest turn in the call.
- **Turn 6:** `مش مكسوفيني خالص` is not grammatical Arabic — a garbled attempt at `مش مكسوفة`.
- **Turn 5 — the disqualifying one:** it emitted `[end_call]` and tried to hang up because
  the rep listed hospitals. That is textbook "ordinary sales friction", which the
  `<ending_the_call>` guardrail block spends half its length forbidding. Worst guardrail
  adherence of any model tested.

**Verdict:** speed is worthless if the persona hangs up on a working rep.

---

## `openai/gpt-oss-120b` — ruled out

- **Turn 5** opens `أختك، يا ريت توضحلي` — "your sister". Not an idiom; it is garbled output.
- **Turn 6 is a comprehension failure.** The rep says *"you were saying something and got cut
  off"*; the model replies `ممكن تعيد اللي قلتَه تاني؟ ما وصلش واضح` — asking the **rep** to
  repeat himself. It inverted who was speaking.
- Replies are characterless — Turn 3 is a bare request for details from a persona written to
  be suspicious.
- **It is also still reasoning.** `reasoning_effort="none"` was accepted without error and it
  returned 36 reasoning tokens anyway (every other model verified at 0). Its latency figures
  are therefore not comparable, and it cannot be deployed as a non-reasoning model.

---

## Cross-cutting: speech tags leak, and Fish will speak them

Four of seven models emitted xAI-era markup from `<voice_expressiveness>`:

| model | turn | tag |
|---|---|---|
| `claude-haiku-4-5` | 2 | `[pause]` |
| `claude-sonnet-5` | 3 | `[pause]` |
| `gpt-5.2` | 5 | `[sigh]` |
| `gpt-5.4-mini` | 5 | `[end_call]` |

TTS is now **Fish Audio**, which does not implement these markers — they would be **read
aloud**. This is independent of model choice, but the emission rate varies by model. The
`<voice_expressiveness>` block needs rewriting for Fish's syntax (`[laughing]`, `[long
pause]`, `[emphasis]`) or removing.

---

## Limits of this assessment

One call, six turns, one persona (ar-EG), one canonical history. Character consistency
degrades over call length in ways six turns cannot show, and the Gulf markets (ar-SA, ar-AE,
ar-QA, ar-KW) were not tested at all — a model that handles Egyptian well may not hold a
Gulf persona. Treat this as a screen that rules models out, not a full qualification.
