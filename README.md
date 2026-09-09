# Delta Twitter support agent

An AI support agent for **@Delta**, built from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
corpus (2.8M tweets). It classifies an incoming public tweet into one of 10 intents,
drafts a reply grounded in how Delta historically handled similar cases, and decides
whether to auto-handle or escalate — with a stated reason.

The system is the easy half. The evaluation is the point, so most of this README is about
how it is measured and where the measurement is weak.

Go 1.22+, **zero third-party dependencies**.

---

## Quickstart

```bash
git clone <this-repo> && cd hiver-support-agent
make fetch-data        # ~493 MB, the raw TWCS csv
make repro             # offline, from the committed LLM cache
```

`make repro` runs with `LLM_PROVIDER=offline`, so every LLM call must hit the committed
cache. It either reproduces the published numbers exactly or fails loudly — it cannot
silently generate different ones. Output lands in **[`reports/RESULTS.md`](reports/RESULTS.md)**
and `results/metrics.json`.

To regenerate from scratch you need a key:

```bash
export GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY, or a local `claude` binary
make all                       # discover intents -> sample -> label -> run -> judge -> report
```

| command | what it does |
|---|---|
| `make repro` | reproduce headline numbers offline |
| `make data` | rebuild Delta episodes from the raw csv |
| `make review` | human annotation console (see [Golden set](#3-golden-set)) |
| `make audit` | blind re-labelling of a random subset |
| `make rate-replies` | blind human reply grading for judge agreement |
| `make test` | unit tests |

---

## 1. Problem framing

### What "good" means for Delta on Twitter

Public Twitter support is **triage, not resolution**. Delta's own agents deflect ~23% of
first replies to DM, and almost nothing is actually resolved in-channel. So the job is not
"answer the customer" — it is *route correctly and say something that does not make things
worse while routing*.

That reframes the objective. The agent is valuable if it can absorb the slice of volume
that genuinely needs no human — praise, channel questions, generic policy — so agents spend
their time on the rest. It is dangerous if it auto-replies to something that needed a person.

So the metric pair is:

- **auto-handle rate** — share of traffic handled with no human. This is the value.
- **false-auto rate** — of the messages it handled alone, how many actually needed a human.
  This is the bill.

Neither number means anything alone. A system that escalates everything scores a perfect
0% false-auto and is worthless; one that auto-handles everything has maximum coverage and is
a liability. Both are shipped as baselines (`trivial-escalate`, `trivial-auto`) so no single
number can be quoted without its cost.

Errors are **asymmetric**. Over-escalating wastes an agent's minute. Auto-replying to a
discrimination complaint, a medical incident or a refund claim is a different category of
failure. The routing policy is built around that asymmetry rather than around accuracy.

### What I deliberately did not build

- **Multi-turn dialogue.** The agent acts on the first message of a thread only. Multi-turn
  needs conversation state and a far harder evaluation; single-turn triage is where the
  volume is.
- **Retrieval over a knowledge base.** There is no Delta policy corpus in the dataset.
  Grounding is against Delta's own past *replies*, which is what the data actually supports.
  A reply is grounded if Delta has demonstrably said something like it before.
- **A fine-tuned classifier as the shipped system.** The trained TF-IDF+logreg classifier
  exists, but as the *baseline*. With 200 labelled examples, fine-tuning would mostly measure
  the golden set.
- **Sentiment/urgency as separate models.** Folded into the escalation decision, where they
  change an action, rather than predicted as unused labels.
- **Actually sending anything.** No Twitter integration. The output is a decision plus a draft.

---

## 2. How it works

```
twcs.csv  ──build-dataset──▶  25,685 Delta episodes
                                    │
                     temporal split (70 / 30)
                     ├─ history 17,979 ──▶ TF-IDF retrieval index
                     └─ eval pool 7,706 ──▶ golden set (200)

incoming tweet
      │
      ├─▶ retrieve k=6 precedents from history (de-duplicated by reply template)
      │
      ├─▶ one LLM call ──▶ { intent, confidence, reply, route, route_reason }
      │
      └─▶ deterministic guardrails ──▶ final route + stated reason
```

**Temporal split, not random.** History is the earliest 70% by timestamp; the eval pool is
the latest 30%. Delta's volume is concentrated in Oct–Dec 2017 and mass-disruption days
produce near-identical tweets minutes apart. Under a random split the retriever would
routinely find a *twin* of the test message — same incident, same hour — and the reply score
would measure duplicate detection rather than generalisation.

**Intents** were derived from the data: k-means over 6,000 historical messages produced 30
clusters (`reports/intent_clusters.json`), which an LLM named and I merged by hand into 10.
Seven of the raw clusters were near-duplicate flavours of "positive feedback". The organising
principle is *what the desk does next*, not topic similarity — two messages share an intent
when they route to the same queue and need the same information.

`flight_disruption` · `baggage` · `booking_change` · `refund_compensation` · `loyalty_program`
· `digital_issue` · `service_complaint` · `contact_support` · `general_inquiry` · `praise` · `other`

**Routing is a one-way ratchet.** The LLM proposes a route; deterministic regex guardrails can
push it towards `escalate` and never away. Eight hard rules (safety/medical, legal,
discrimination, vulnerable passenger, money claim, account/PII, explicit human request, press
risk) plus two confidence gates (low intent confidence, no usable precedent) plus the intent's
own default. This is enforced in `internal/agent/policy.go` and asserted in tests — not left
to the prompt, because a prompt is not a control.

---

## 3. Golden set

**200 hand-adjudicated examples**, drawn from the eval pool in two strata that are *never
pooled for a headline claim*:

| stratum | n | how | what it measures |
|---|---:|---|---|
| `random` | 120 | uniform draw | unbiased estimate of live performance |
| `targeted` | 80 | round-robin over guardrail matches, then 20 topical clusters | whether it breaks where breaking is expensive |

Labelling protocol (full detail in **[`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md)**,
generated from the source so it cannot drift from the rules the annotator saw):

1. **Two independent draft passes** — A reasons down from the taxonomy, B up from the
   customer's need. Both see the **full thread**, including Delta's actual replies and whether
   the customer came back angry. That is oracle information the system under test never gets,
   which is what makes these labels a gold standard rather than a second opinion on the same task.
2. **Adjudication** — a third pass resolves disagreements. Raw A/B agreement is kept as a
   per-item difficulty signal.
3. **Human review** (`make review`) — every item shown with its draft; confirm or override.
4. **Blind audit** (`make audit`) — a random subset re-labelled cold, draft hidden.

Step 3 is **anchored**: the reviewer saw the draft, so its override rate is a lower bound on
true disagreement. Step 4 is the number the report quotes as evidence of label quality.

> **Provenance, stated plainly.** The draft labels are model-generated. A human adjudicates
> them via `make review`; the report distinguishes model-drafted from human-confirmed items and
> reports the override rate. If the review pass has not been run, the report says so rather
> than presenting drafts as gold.

---

## 4. Evaluation harness

**Automated metrics.** Intent accuracy and macro-F1 (averaged only over labels present in the
gold set) with bootstrap CIs; per-class P/R/F1; confusion matrix. Routing: auto rate, false-auto
rate, escalation recall, over-escalation, each with a Wilson 95% interval — at n=200 with rare
events the normal approximation is badly wrong.

**LLM-as-judge.** A four-part rubric — `grounding` (1–5), `resolution` (1–5), `tone` (1–5),
`sendable` (would a Delta social lead post this unedited), plus a free-text `violation` field.
Composite = 0.5·grounding + 0.3·resolution + 0.2·tone, weighting grounding highest because an
ungrounded reply creates a promise Delta has to honour.

Three deliberate choices:

- The judge **never sees which system** wrote a reply.
- The judge runs on a **stronger model than the agent** — a judge sharing the agent's weights
  shares its blind spots and rates its own output generously.
- Delta's real reply is shown as **one acceptable answer, not ground truth**, and the judge is
  told explicitly that Delta's own replies are often lazy deflections that would score poorly.
  Otherwise "please DM us" becomes the optimal output.

**Judge-vs-human agreement.** `make rate-replies` presents a blind sample — system hidden,
judge's scores hidden — for a human to grade on the same rubric. The report gives exact
agreement, within-1, MAE, **quadratic-weighted κ** (4-vs-5 is not the same failure as 1-vs-5),
Spearman, and the judge's bias sign. It separately reports agreement on the binary ship/no-ship
call, because a judge that matches on scores but not on that call is useless for the decision
the metric exists to support.

**Delta's own replies are graded by the same judge**, as a reference row. Beating a baseline
says the agent is better than nothing; comparing against the human desk says whether it is good
enough to deploy.

---

## 5. Results

> **Not yet generated.** The pipeline is complete and tested, but the annotation, agent and
> judge runs require an LLM key. Run `make all` (or `make repro` once the cache is committed)
> and every table below is written to [`reports/RESULTS.md`](reports/RESULTS.md).

`reports/RESULTS.md` contains:

1. Evaluation-set composition and label provenance
2. Intent classification vs both baselines, with CIs, per-class F1 and a confusion matrix
3. Routing metrics, reported **separately per stratum**
4. Reply quality incl. the `delta-human` reference row and `auto-sendable`
5. Judge-vs-human agreement
6. Label quality — anchored *and* blind
7. Guardrail ablation: model-only routing vs shipped, and per-rule firing precision
8. Confidence calibration
9. Failure gallery with real examples

**Systems compared:**

| system | intent | reply | routing |
|---|---|---|---|
| `trivial-escalate` | majority class | canned "DM us" | always escalate |
| `trivial-auto` | majority class | canned "DM us" | always auto |
| `simple` | TF-IDF + logreg, 5-fold CV | verbatim copy of nearest historical reply | rules only, no LLM |
| `agent-no-retrieval` | LLM | LLM, no precedents | LLM + rules |
| `agent` | LLM | LLM grounded on 6 precedents | LLM + rules |
| `delta-human` | — | what Delta actually sent | — |

The trivial baseline's canned reply is not a straw man: ~23% of Delta's real first replies are
a variant of "please DM us", so it is a genuine competitor on any reply-similarity metric —
which is exactly what makes it useful.

The `simple` baseline is deliberately strong. It copies real Delta replies verbatim, so it
*cannot hallucinate*, and its classifier is trained on the golden set under cross-validation —
labelled data the zero-shot agent never sees. That handicaps the agent on purpose.

---

## 6. What is misleading about my headline number

A mandatory section, and the honest answers are uncomfortable.

**1. The gold labels are model-drafted.** A human adjudicates them, but the drafts come from
the same family of model being evaluated. Shared blind spots inflate every accuracy number by
an unknown amount. The blind audit bounds this, but on a subset — and the reviewer is one
person, not a panel, so there is no inter-human agreement number at all.

**2. The judge and the agent are correlated.** Different models, but similar training data and
similar failure modes. LLM judges are known to prefer LLM-written text over human-written text;
the `delta-human` reference row exists partly to expose this. If the judge scores the agent
above Delta's own staff, treat that as evidence about the judge, not the agent.

**3. n=200 makes the interesting cells tiny.** False-auto rate is computed over only the
messages a system auto-handled. If the agent auto-handles 30% of 120 random-stratum items,
that denominator is ~36, and one error moves the rate by three points. The Wilson intervals in
the report are wide for exactly this reason. **Read the intervals, not the point estimates.**

**4. Pooled numbers overstate difficulty; random-stratum numbers understate rare risk.** The
targeted stratum is escalation-heavy by construction and is not a traffic estimate. The random
stratum is unbiased but contains too few safety/legal cases to say anything reliable about the
worst failure mode. Neither stratum answers the question alone, which is why they are never merged.

**5. "Auto-handle rate" is measured against my own routing rule, which I also wrote.** The
gold `route` label follows a policy I authored. A different support organisation would draw the
auto/escalate line elsewhere, and every routing number would move. The metric measures
consistency with a stated policy, not correctness in any absolute sense.

**6. Guardrail precision is partly circular.** The regex rules and the annotation guideline
both encode "money and safety need a human". Rules firing on cases the gold label also marked
escalate is partly the same judgement measured twice.

**7. One brand, one channel, 2017.** Delta on Twitter in late 2017. Nothing here transfers to
another brand without redoing the taxonomy, and the language of a 2017 tweet is not the language
of a 2026 support message.

**8. Cost and latency are unmeasured.** Every quality number ignores that the agent makes an LLM
call per message. A desk deciding whether to deploy needs cost per handled ticket, which this
does not provide.

---

## 7. What I would do next, given one more week

1. **A second human annotator** on 100 items, to get a real inter-annotator agreement ceiling.
   Every accuracy number is currently uninterpretable without it — you cannot tell a model at 80%
   from a task where humans only agree 80% of the time.
2. **Calibrate the escalation threshold against a cost model.** Ask Delta what an over-escalation
   costs versus a false-auto, then pick the operating point on a coverage/risk curve instead of
   hard-coding 0.60.
3. **Replace TF-IDF retrieval with embeddings** and measure whether it matters. The current
   retriever fails on paraphrase ("bag never showed up" vs "luggage missing"). The ablation
   harness is already in place to answer this cleanly.
4. **Adversarial judge validation** — feed the judge deliberately corrupted replies (invented
   flight numbers, fabricated compensation) and check it catches them. Agreement with a human on
   ordinary replies does not prove it detects the failure it exists to detect.
5. **Escalation reason quality.** The agent states a reason for every decision, and nothing
   currently evaluates whether that reason is *true*. A plausible-but-wrong reason is worse than none.
6. **Prevalence-weighted reporting** to combine the two strata into one estimate with a
   principled weight, instead of reporting them side by side.

---

## 8. Repo layout

```
cmd/
  build-dataset/     twcs.csv -> per-brand episodes
  discover-intents/  clustering + LLM naming -> taxonomy proposal
  sample-golden/     two-stratum sampling from the eval pool
  label-assist/      two-pass + adjudication draft labels
  review/            human console: labels | audit | replies
  run-agent/         all five systems over the golden set
  judge/             LLM judge + blind human rating tasks
  report/            metrics -> reports/RESULTS.md + results/metrics.json
  gen-docs/          regenerates docs/ANNOTATION_GUIDE.md from source
internal/
  data/       episode extraction, cleaning, temporal split
  textx/      tokenizer, TF-IDF, inverted index, spherical k-means
  ml/         multinomial logistic regression, stratified folds
  taxonomy/   the 10 intents + the route-labelling policy
  agent/      retriever, escalation policy, agent pipeline
  baseline/   trivial and simple systems
  eval/       metrics, Wilson/bootstrap CIs, judge, ordinal agreement
  llm/        gemini | anthropic | cli providers + committed disk cache
```

**[`reports/DECISIONS.md`](reports/DECISIONS.md)** — the 17 non-obvious decisions and why.

---

## 9. Notes on the LLM layer

Three providers behind one interface: `gemini` (`GEMINI_API_KEY`), `anthropic`
(`ANTHROPIC_API_KEY`), and `cli` (a local `claude` binary). Selected via `LLM_PROVIDER`, else
inferred from whichever key is present.

Every completion is cached to `cache/llm_cache.jsonl`, keyed by a hash of the request —
**not** the provider — so a cache built with one provider is reused by another. The cache is
committed. That is what makes `make repro` run offline in minutes, and it is why a reviewer
does not need a key to check the numbers.

Gemini's default safety filters are disabled for this workload. Real support tweets contain
profanity and descriptions of conflict; filtering them would silently bias every metric towards
the polite half of the corpus. Anything still blocked surfaces as an error rather than a
silent drop.

---

## 10. Citations and borrowings

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (Kaggle, `thoughtvector`). Fetched via the [`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs)
  Hugging Face mirror, which hosts the identical `twcs.csv`, because Kaggle needs credentials.
- **TF-IDF formulation** mirrors scikit-learn's `TfidfVectorizer` defaults — smoothed IDF
  `ln((1+n)/(1+df)) + 1`, sublinear TF, L2 norm — so the numbers are comparable to a standard
  implementation. Written from scratch in `internal/textx`; no code copied.
- **k-means++ seeding**: Arthur & Vassilvitskii (2007).
- **Wilson score interval**: Wilson (1927), used instead of the normal approximation because
  the event counts here are small.
- **Cohen's κ** (1960) and **quadratic-weighted κ** (Cohen 1968) for nominal and ordinal
  agreement respectively.
- **LLM-as-judge design** — pointwise scoring to avoid position bias, a judge stronger than
  the system under test, and blinding to system identity — follows the practices established
  in the MT-Bench / "LLM-as-a-Judge" line of work (Zheng et al., 2023).
- Everything else is written for this repo. Claude Code was used as a coding assistant
  throughout; the design decisions, taxonomy, routing policy and evaluation methodology are mine.
