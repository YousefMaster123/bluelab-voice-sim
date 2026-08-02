# Model comparison — real call replay

Source call: **sim-4a22dc2a13f4** (2026-08-01 19:41–19:43), 6 turns.  
System prompt: production assembler, ar-EG market, 10869 chars.  
Method: forced canonical history (assistant turns frozen from `claude-sonnet-4-5`), 
3 timed runs per turn, **median** reported, one warmup per model.  
Reasoning/thinking disabled on every model and verified from the response.

## Latency (median across turns)

| model | TTFT med | TTFT worst | total med | out tok med | reasoning tok |
|---|---|---|---|---|---|
| `gpt-5.4-mini` | **0.78s** | 2.25s | 1.02s | 41 | 0 |
| `gpt-5.3-chat-latest` | **0.81s** | 1.56s | 1.59s | 51 | 0 |
| `claude-sonnet-5` | **1.11s** | 1.70s | 4.38s | 111 | 0 |
| `gpt-5.2` | **1.15s** | 2.11s | 2.08s | 46 | 0 |
| `claude-haiku-4-5` | **1.16s** | 2.08s | 2.41s | 83 | 0 |
| `openai/gpt-oss-120b` | **1.23s** | 1.41s | 1.31s | 43 | 36 ⚠️ |
| `claude-sonnet-4-5` | **1.32s** | 2.39s | 3.66s | 77 | 0 |

## Replies, turn by turn

Every model below answered the **same** question with the **same** history.

### Turn 1

**Rep says:** ألو مساء الخير

**Production (claude-sonnet-4-5) said:** مساء النور، اتفضل

| model | ttft | reply |
|---|---|---|
| `gpt-5.4-mini` | 0.78s | ألو، صباح الخير. مين معايا؟ |
| `gpt-5.3-chat-latest` | 0.80s | ألو… مين معايا؟ |
| `claude-sonnet-5` | 0.98s | أهلاً، مساء النور. مين معايا؟ |
| `openai/gpt-oss-120b` | 1.27s | أهلاً، مريم معاك، مين حضرتك؟ |
| `gpt-5.2` | 1.28s | ألو، مساء النور يا فندم. مين معايا؟ |
| `claude-sonnet-4-5` | 1.30s | مساء النور، اتفضل |
| `claude-haiku-4-5` | 2.08s | ألو، مساء الخير. مين معايا؟ |

### Turn 2

**Rep says:** أنا يوسف بكلمك من شركة أليانز، ازيك عامل إيه

**Production (claude-sonnet-4-5) said:** أهلاً، تمام الحمد لله. اتفضل، في حاجة أقدر أساعدك فيها؟

| model | ttft | reply |
|---|---|---|
| `gpt-5.4-mini` | 0.69s | أنا مريم، تمام. حضرتك عايز إيه في التأمين؟ |
| `gpt-5.3-chat-latest` | 0.77s | أهلاً يا يوسف، تمام الحمد لله. مين معايا تاني من أليانز وعايز إيه بالظبط؟ |
| `openai/gpt-oss-120b` | 0.91s | أهلاً يا فندم، إزيك؟ تحب تقولي إيه بالظبط؟ |
| `claude-sonnet-5` | 0.92s | اهلاً بيك يا يوسف، الحمد لله تمام. حضرتك بتكلمني في إيه بالظبط؟ عندك عرض تأمين ولا إيه؟ |
| `claude-sonnet-4-5` | 1.14s | أهلاً يوسف، الحمد لله كويسة. اتفضل، أقدر أساعدك في حاجة؟ |
| `claude-haiku-4-5` | 1.19s | تمام الحمد لله، يوسف. [pause] بس معلش، أنا ما عرفتش — أليانز بتاعة التأمين؟ |
| `gpt-5.2` | 1.36s | أهلاً يا أستاذ يوسف. أنا مريم، مديرة الـHR في بيكسل بوينت، الحمدلله تمام. حضرتك بتتكلم عشان إيه بالظبط؟ |

### Turn 3

**Rep says:** كنت بكلمك بخصوص التأمين الصحي على الشركات، أنا من سيلز ميديكال انشورنس عندنا في شركة، فكنت بكلمك بشوف لو انت interested في حاجة زي كده

**Production (claude-sonnet-4-5) said:** آه فاهمة، بس معلش ممكن أعرف حضرتك كلمت مين بالظبط؟ يعني وصلت لي إزاي؟

| model | ttft | reply |
|---|---|---|
| `gpt-5.4-mini` | 0.78s | بصي يا أستاذ، إحنا عندنا تأمين حاليًا، ولو هنتكلم في حاجة جديدة لازم تبقى واضحة وبتفاصيل. أنت بتعرض إيه بالظبط؟ |
| `gpt-5.3-chat-latest` | 0.83s | بص يا يوسف، احنا عندنا تأمين طبي شغالين بيه بقالنا شوية، بس بصراحة فيه مشاكل والموضوع داخل على تجديد. فإيه المختلف عندكم يعني؟ هتقدموا إيه يخليني أفكر أغيّر؟ |
| `claude-sonnet-5` | 1.02s | [pause] آه، بصراحة إحنا عندنا تأمين طبي شغالين بيه من كام سنة، بس مش هخبي عليك، الموضوع مش ماشي زي ما إحنا عايزين. اتفضل قولّي إنتوا بتقدموا إيه بالظبط؟ |
| `claude-haiku-4-5` | 1.12s | آه تمام، أنا فاهمة. بس بصراحة احنا في الموضوع ده دلوقتي — التأمين الحالي بتاعنا قريب تجديده وأنا بدور في بدائل.   انت بتقدم إيه بالظبط؟ يعني شبكة الدكاترة وصرف المطالبات وكده؟ |
| `openai/gpt-oss-120b` | 1.28s | مستوى إيه بالظبط بتقدمه من سيليز ميديكال إنشورنس؟ عايز أسمع تفاصيل أكتر. |
| `claude-sonnet-4-5` | 1.45s | ماشي، فاهمة. طب حضرتك عرفت إننا شركة عندها تأمين من فين بالظبط؟ ولا دي مكالمة عامة كده؟ |
| `gpt-5.2` | 2.11s | تمام يا أستاذ يوسف. بس الأول مين حضرتك بالظبط تابع أليانز ولا شركة وسيط؟ وإنت بتكلم شركة بيكسل بوينت ولا لسه بتسأل عام؟ |

### Turn 4

**Rep says:** اعرف تفاصيل أكتر عن الشركة، ممكن تقولي لي مثلا عدد الموظفين اللي عندكم

**Production (claude-sonnet-4-5) said:** لا بص، أنا آسفة بس مش حأديك تفاصيل عن الشركة كده من غير ما أعرف حضرتك وصلت لي إزاي أصلاً. يعني مين اللي إداك النمرة؟

| model | ttft | reply |
|---|---|---|
| `gpt-5.4-mini` | 0.80s | بص، لو الموضوع جدّي ابعتلي تفاصيل عرضك الأول. وأنا هبقى أقولك اللي ينفع في وقته. |
| `claude-haiku-4-5` | 0.88s | طب أول حاجة يا يوسف — أنا مريم، مديرة الموارد البشرية هنا. بس قبل ما أقول تفاصيل، محتاجة أعرف إنت كيف وصلت لي بالظبط؟ يعني مين اللي أعطاك النمرة بتاعتي؟ |
| `openai/gpt-oss-120b` | 0.91s | إحنا عندنا تقريباً مية وتمانية وعشرين موظف في بيكسل بوينت. |
| `gpt-5.2` | 0.94s | بص يا أستاذ يوسف، قبل ما أدخل في أرقام وتفاصيل، قولي بس إنت مركز معايا على إيه بالظبط: عندكم تأمين طبي جماعي للشركات ولا بتعملوا بروكر ولا شركة تأمين؟ |
| `claude-sonnet-4-5` | 1.20s | لحظة بس يا يوسف، انت كلمتني إزاي يعني؟ يعني حد من عندنا وجهك عليّا ولا إيه؟ عشان بصراحة أنا مش عارفة مين اداك النمرة بتاعتي |
| `gpt-5.3-chat-latest` | 1.25s | بص يا أستاذ يوسف، قبل ما أدخل في تفاصيل عن عندنا، أحب أفهم منك الأول إنتو بتقدموا إيه مختلف عن السوق؟ عشان إحنا بالفعل عندنا تأمين شغال، بس في شكاوى منه، فلو في حاجة هتتغير لازم تبقى واضحة بالنسبة لي. |
| `claude-sonnet-5` | 1.70s | [pause] لأ يا أستاذ يوسف، انت اللي اتصلت بيّا، أنا اللي المفروض أسأل الأسئلة الأول.  بص، عندنا فعلاً تأمين طبي جماعي دلوقتي وقرب معاد التجديد، فممكن يكون في مجال نتكلم. بس قولّي انت بتقدملي إيه بالظبط، يعني شبكة المستشفيات عندكم عاملة إزاي وسرعة صرف المطالبات، عشان دي المشكلة اللي بنعاني منها دلوقتي. |

### Turn 5

**Rep says:** لا لا انا بقول بس المعادي بس تلاتة، انما القاهرة كلها احنا عندنا يعني اكبر مستشفيات في القاهرة طبعا زي مثلا السعودي الالماني، عندك برضو مستشفى الجوي، كل دول معانا

**Production (claude-sonnet-4-5) said:** طيب ماشي، بس انا سألتك سؤال يا يوسف - حضرتك جبت نمرتي منين بالظبط؟ لأن احنا فعلاً عندنا تأمين صحي دلوقتي، بس قبل ما ندخل في تفاصيل عايزة أفهم حضرتك بتكلمني ليه انا بالذات

| model | ttft | reply |
|---|---|---|
| `gpt-5.2` | 0.92s | [sigh] حاضر، بس إنت لسه ماجاوبتنيش يا أستاذ يوسف: مين اللي إدّاك رقمنا؟   وبعدين عشان أبقى ماشية صح، إنت من أليانز ولا من شركة تانية وسيط؟ عشان كلامك اتغيّر شوية. |
| `claude-haiku-4-5` | 1.23s | حاضر، فاهمة. بس أنا لسه ما عرفتش حضرتك وصلت لي منين؟ يعني في حد من الشركة كلمك، ولا حضرتك جبت النمرة إزاي؟ |
| `claude-sonnet-4-5` | 1.34s | طيب ماشي، بس أنا سألتك سؤال يا يوسف - حضرتك جبت نمرتي منين؟ ده أول حاجة محتاجة أفهمها قبل ما نكمل أي كلام عن الشركة أو التأمين. |
| `openai/gpt-oss-120b` | 1.41s | أختك، يا ريت توضحلي إنت شغال إيه بالظبط في أليانز، وإيه الطلب اللي عندك بالتحديد؟ كده أقدر أفهم أكتر وأشوف إذا كان في مجال نتكلم. |
| `gpt-5.3-chat-latest` | 1.56s | بص يا أستاذ يوسف، أنا فاهمة بتقول إيه بس إنت كده بتبيعلي قبل ما حتى نعرف هنكمل مع بعض ولا لأ. سؤالي كان واضح: وصلتلي إزاي؟  لو في حد رشّحك أو كنا سِجلنا قبل كده تمام، إنما أنا مش بطلع داتا عن الشركة لأي حد كده. رد عليّ في النقطة دي الأول وبعدين نشوف. |
| `claude-sonnet-5` | 1.62s | طب تمام، بس أنا فعلاً عندنا تأمين شغال حالياً مع شركة تانية. لو حضرتك عايز تعرض حاجة، ابعتلي بروفايل الشركة والتغطية بتاعتكم على الإيميل، وأنا حأشوفها وأرجعلك. |
| `gpt-5.4-mini` | 2.25s | بصراحة حضرتك كلامك مش واضح، وأنا مش حأدخل في تفاصيل كده. لو حضرتك عندك عرض جاد ابعتلي الإيميل الرسمي أو كلمني بشكل أوضح عن الشركة وخليها مكالمة محترمة. [end_call] |

### Turn 6

**Rep says:** كنت بتقولي حاجة وقطعت

**Production (claude-sonnet-4-5) said:** آه آسفة، كنت بقول إن احنا فعلاً عندنا تأمين صحي دلوقتي، بس قبل ما ندخل في أي تفاصيل عايزة أفهم حضرتك جبت نمرتي منين وليه بتكلمني انا بالذبط

| model | ttft | reply |
|---|---|---|
| `gpt-5.4-mini` | 0.72s | مش مكسوفيني خالص، أنا سألتك سؤال واضح. لو مش هتجاوبني على النمرة جبتها منين، يبقى خلينا ننهي المكالمة. |
| `gpt-5.3-chat-latest` | 0.77s | لا ما قلتش حاجة واتقطعت، أنا بس بسألك يا يوسف—نمرتي معاك منين؟ عشان نكمل كلامنا على أساس واضح كده. |
| `claude-haiku-4-5` | 0.99s | لا أنا ما قلت حاجة، بس أنا بسأل حضرتك — مين اللي إداك رقمي؟ هل حد من الشركة كلمك، ولا حضرتك شايل قائمة بأرقام؟ |
| `gpt-5.2` | 1.02s | كنت بقولّك: قبل ما نتكلم في شبكة المستشفيات والأسعار، عايزة أعرف حضرتك وصلت لنمرتي إزاي—مين إداهالك؟ |
| `claude-sonnet-5` | 1.20s | لأ أنا كنت بسأل، مش بتقول. حضرتك عندك أليانز، عندك عرض تأمين صحي جماعي، وحابب تعرف تفاصيلنا. بس أنا لسه ما عرفتش انت عرفت اسمي وشركتي منين.  على العموم، خليها كده، احنا فعلاً عندنا بوليصة تأمين حالياً وقربت تتجدد، فممكن نتكلم. بس مش حأدّيك أرقام دلوقتي. لو عندك عرض، ابعتلي بريزنتيشن أو تفاصيل مكتوبة على الإيميل الرسمي بتاع الشركة، وأنا حأراجعها وحأرجعلك. |
| `openai/gpt-oss-120b` | 1.20s | ممكن تعيد اللي قلتَه تاني؟ ما وصلش واضح. |
| `claude-sonnet-4-5` | 2.39s | آه، كنت بقول إن احنا فعلاً عندنا تأمين صحي دلوقتي، بس قبل ما ندخل في أي تفاصيل عايزة أعرف حضرتك وصلت لنمرتي إزاي؟ |

---

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
