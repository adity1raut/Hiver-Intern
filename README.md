# Delta Twitter support agent

Triage agent for @Delta built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
corpus. For each inbound tweet it assigns one of 10 intents, drafts a reply grounded in
Delta's own reply history, and decides auto-handle vs escalate with a stated reason.

Go 1.22, no third-party dependencies.

## Running it

```
make fetch-data      # 493 MB twcs.csv
make repro           # offline, from the committed cache
```

`repro` sets `LLM_PROVIDER=offline`, so a cache miss is a hard error. It reproduces the
committed numbers or it fails. Results go to `reports/RESULTS.md` and `results/metrics.json`.

Regenerating from scratch needs a key:

```
export GEMINI_API_KEY=...     # or ANTHROPIC_API_KEY, or a local claude binary
make all
```

| target | |
| --- | --- |
| `make data` | rebuild episodes from the raw csv |
| `make labels` | draft golden labels |
| `make review` | human annotation console |
| `make audit` | blind re-label of a random subset |
| `make rate-replies` | blind human reply grading |
| `make test` | 31 unit tests |

## Framing

Public Twitter support is triage. Delta deflects about 23% of first replies to DM and
resolves almost nothing in-channel, so the job is to route correctly and not make things
worse while routing. The agent earns its place if it absorbs the volume that needs no human
at all: praise, channel questions, generic policy. It is a liability if it answers something
that needed a person.

That gives two numbers that have to be read together:

* auto-handle rate: share of traffic closed without a human.
* false-auto rate: of those, how many actually needed one.

Either one alone is gameable. Escalate everything and false-auto is 0% and the agent is
useless. Auto-handle everything and coverage is 100% and it is dangerous. Both extremes ship
as baselines (`trivial-escalate`, `trivial-auto`) so neither number can be quoted alone.

The errors are not symmetric. Over-escalating costs an agent a minute. Auto-replying to a
discrimination complaint or a refund claim is a different kind of failure. The routing policy
is built around that, not around accuracy.

### Not built

* Multi-turn dialogue. The agent sees only the first message of a thread. Multi-turn needs
  conversation state and a much harder eval, and the volume is in triage.
* Retrieval over a knowledge base. There is no Delta policy corpus in the dataset. Grounding
  is against Delta's past replies, which is what the data supports.
* A fine-tuned classifier as the shipped system. It exists, as the baseline. With 200 labels,
  fine-tuning would mostly measure the golden set.
* Separate sentiment or urgency models. Folded into the escalation decision where they change
  an action.
* Any Twitter integration. Output is a decision plus a draft.

## Pipeline

```
twcs.csv -> 25,685 Delta episodes
              |
      temporal split 70/30
      history 17,979  -> TF-IDF retrieval index
      eval pool 7,706 -> golden set (200)

inbound tweet
   -> retrieve k=6 precedents, de-duplicated by reply template
   -> one LLM call: {intent, confidence, reply, route, route_reason}
   -> deterministic guardrails -> final route + reason
```

The split is by timestamp, not random. Delta's volume sits in Oct-Dec 2017 and a mass
cancellation produces near-identical tweets minutes apart. Under a random split the retriever
finds a twin of the test message from the same incident, and the reply score measures
duplicate detection instead of generalisation.

Intents came from the data: k-means over 6,000 historical messages gave 30 clusters
(`reports/intent_clusters.json`), an LLM named them, and I merged them by hand to 10. Seven of
the raw clusters were variants of positive feedback. The merge criterion is what the desk does
next, not topic similarity: two messages share an intent if they hit the same queue and need
the same information.

```
flight_disruption  baggage  booking_change  refund_compensation  loyalty_program
digital_issue  service_complaint  contact_support  general_inquiry  praise  other
```

Routing is a one-way ratchet. The LLM proposes a route; the guardrails can only move it toward
escalate. Eight regex rules (safety, legal, discrimination, vulnerable passenger, money, PII,
explicit human request, press risk), two confidence gates, and the intent's own default. This
lives in `internal/agent/policy.go` and is asserted in tests rather than left to the prompt.

## Golden set

200 items from the eval pool, in two strata that are never pooled for a headline claim.

| stratum | n | how | reads as |
| --- | --- | --- | --- |
| random | 120 | uniform | estimate of live performance |
| targeted | 80 | guardrail matches, then 20 topical clusters | behaviour where failure is expensive |

Labelling, in full in [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md), which is generated
from the same constants the prompts use:

1. Two independent draft passes. A works down from the taxonomy, B up from the customer's need.
   Both see the whole thread including Delta's replies and whether the customer came back. The
   system under test never sees that.
2. A third pass adjudicates disagreements. Raw A/B agreement is kept as a difficulty signal.
   It came out at 84.5%, so 31 of 200 needed adjudication.
3. `make review`: every item shown with its draft, confirm or override.
4. `make audit`: a random subset re-labelled cold with the draft hidden.

Step 3 is anchored, so its override rate is a floor on real disagreement. Step 4 is the number
worth quoting.

The drafts are model-generated. The report reports how many items a human confirmed and how
many it changed, and says so plainly when review has not been run.

## Evaluation

Intent accuracy and macro-F1 over labels present in the gold set, with bootstrap CIs, per-class
P/R/F1 and a confusion matrix. Routing: auto rate, false-auto rate, escalation recall,
over-escalation, each with a Wilson interval. At n=200 the interesting denominators are small
enough that the normal approximation is wrong.

The judge scores grounding, resolution and tone on 1-5, plus a binary sendable call and a
free-text violation field. Composite is 0.5/0.3/0.2, grounding weighted highest because an
ungrounded reply creates a promise Delta has to keep.

Three things about the judge setup:

* It never sees which system wrote a reply.
* It runs on a larger model than the agent. A judge sharing the agent's weights shares its
  blind spots.
* Delta's real reply is shown as one acceptable answer, not as ground truth, and the judge is
  told Delta's own replies are often deflections that would score badly. Otherwise "please DM
  us" becomes the optimal output.

Delta's own replies are graded on the same rubric as a reference row. Beating a baseline says
the agent beats nothing. The reference row says whether it is near the desk it would join.

`make rate-replies` shows a human a blind sample with the system and the judge's scores hidden.
The report gives exact agreement, within-1, MAE, quadratic-weighted kappa, Spearman and the
judge's bias, plus separate agreement on the sendable call. A judge that tracks the scores but
not the ship/no-ship decision is no use for the decision the metric exists for.

## Results

Full tables in [`reports/RESULTS.md`](reports/RESULTS.md), regenerated by `make report`.
Agent is Haiku 4.5, judge is Sonnet 5, annotation is Sonnet 5.

| system | intent | reply | routing |
| --- | --- | --- | --- |
| trivial-escalate | majority class | canned DM deflection | always escalate |
| trivial-auto | majority class | canned DM deflection | always auto |
| simple | TF-IDF + logreg, 5-fold CV | verbatim nearest historical reply | rules only |
| agent-no-retrieval | LLM | LLM, no precedents | LLM + rules |
| agent | LLM | LLM on 6 precedents | LLM + rules |
| delta-human | | what Delta sent | |

Intent classification, n=200:

| system | accuracy | 95% CI | macro-F1 |
| --- | --- | --- | --- |
| trivial | 25.5% | 19-32 | 0.04 |
| simple | 30.5% | 24-36 | 0.12 |
| agent-no-retrieval | 83.0% | 78-88 | 0.80 |
| agent | 81.0% | 75-86 | 0.77 |

Routing, random stratum only, which is the estimate of live behaviour:

| system | auto rate | false-auto | 95% CI | escalation recall |
| --- | --- | --- | --- | --- |
| trivial-escalate | 0% | 0% | 0-0 | 100% |
| trivial-auto | 100% | 43.3% | 35-52 | 0% |
| simple | 15.8% | 15.8% | 6-38 | 94.2% |
| agent-no-retrieval | 45.0% | 5.6% | 2-15 | 94.2% |
| agent | 45.8% | 3.6% | 1-12 | 96.2% |

Reply quality, judge composite out of 5:

| system | composite | sendable | auto-sendable | violations |
| --- | --- | --- | --- | --- |
| simple | 3.06 | 6.0% | 24.0% (n=25) | 69% |
| trivial-escalate | 3.49 | 5.5% | | 56.5% |
| delta-human | 3.50 | 23.0% | | 46% |
| agent | 3.91 | 25.5% | 26.7% (n=75) | 31% |
| agent-no-retrieval | 3.93 | 32.5% | 38.4% (n=73) | 25% |

The guardrails do most of the routing work:

| policy | auto rate | false-auto |
| --- | --- | --- |
| model's own judgement alone | 70.5% | 29.8% |
| plus guardrails, as shipped | 37.5% | 2.7% |

They flipped 66 of the model's `auto` calls to `escalate`, an eleven-fold cut in false-auto
rate. Rule precision against the gold route runs from 1.00 (`legal_or_regulatory`,
`discrimination_or_conduct`) down to 0.33 (`low_intent_confidence`).

The trivial reply is not a straw man. About 23% of Delta's real first replies are some form of
"please DM us", so it competes honestly on any reply-similarity metric, which is the point. It
scores 4.56 on grounding, second highest of anything measured, because a reply that says
nothing cannot say anything wrong.

The simple baseline is meant to be hard to beat. It copies real Delta replies word for word so
it cannot hallucinate, and its classifier trains on the golden set under cross-validation,
which is labelled data the zero-shot agent never gets. It still lands at 30.5% intent accuracy,
because copying a reply from a lexically similar tweet is not the same as understanding it.

### Three results that did not go the way I expected

**Retrieval does not earn its place.** The ablation without precedents beats the shipped agent
on intent accuracy (83.0 vs 81.0), composite (3.93 vs 3.91) and sendable rate (32.5% vs 25.5%).
None of those gaps clear their confidence intervals, so the honest reading is that retrieval
makes no difference here, not that removing it helps. Either way the grounding story the
system was built around is not supported by its own ablation. The likely cause is that TF-IDF
retrieves lexically similar tweets rather than similar situations, and Delta's replies are
formulaic enough that six of them mostly teach the model to deflect. This is the first thing
I would fix.

**The judge rates the agent above Delta's own staff** (3.91 vs 3.50 composite, 25.5% vs 23.0%
sendable). I do not believe the agent is better than Delta's support desk. This is the
self-preference bias that the `delta-human` reference row exists to expose, and it means the
absolute judge scores should not be read as quality. The comparison between systems is still
useful because they are graded identically; the comparison against the human row is not.

**Self-reported confidence is not calibrated.** Accuracy by confidence bucket runs 75.0%,
57.6%, 77.4%, 87.5%, 93.5%. It is not monotonic, so the `low_intent_confidence` gate at 0.60
is close to decoration. It fired 3 times in 200 with precision 0.33. Either the threshold
moves to 0.85, where the signal actually appears, or the gate comes out and is replaced by the
logistic regression's probability, which is at least a real posterior.

## What is misleading about the headline number

1. The gold labels are model drafts with a human adjudicating. They come from the same family
   of model under evaluation, so shared blind spots inflate every accuracy figure by an unknown
   amount. The blind audit bounds this on a subset. There is one reviewer, so there is no
   inter-human agreement figure at all.

2. Judge and agent are correlated. Different sizes, similar training data, similar failure
   modes. LLM judges are known to prefer LLM text over human text. The `delta-human` row is
   partly there to expose that. If the judge puts the agent above Delta's own staff, that is
   evidence about the judge.

3. n=200 makes the cells that matter small. False-auto rate is computed only over what a system
   auto-handled. If the agent auto-handles 30% of the 120 random-stratum items, the denominator
   is around 36 and one error moves the rate three points. Read the intervals, not the point
   estimates.

4. Pooled numbers overstate difficulty and random-stratum numbers understate rare risk. The
   targeted stratum is escalation-heavy by construction and is not a traffic estimate. The
   random stratum is unbiased and contains too few safety and legal cases to say much about the
   worst failure mode. Neither answers alone, which is why they are never merged.

5. Auto-handle rate is measured against a routing rule I wrote. The gold route label follows a
   policy I authored. Another support org draws that line elsewhere and every routing number
   moves. This measures consistency with a stated policy, not correctness.

6. Guardrail precision is partly circular. The regexes and the annotation guideline both encode
   "money and safety need a human". A rule firing on a case the gold label also escalated is
   partly the same judgement counted twice.

7. One brand, one channel, late 2017. Nothing transfers without redoing the taxonomy, and 2017
   tweet register is not 2026 support register.

8. Cost and latency are unmeasured. Every quality number ignores one LLM call per message. A
   deployment decision needs cost per handled ticket and this does not provide it.

9. The auto-handle rate is capped by my own taxonomy, not by the model. 23 of the 200 gold
   items are labelled `auto` while their intent's default action is `escalate`, so
   `intent_default_escalate` forces a wrong answer on 11.5% of the set no matter what the
   model does. Maximum achievable auto rate is 43.5%, not 55%, and at least 21% of the
   over-escalation is structural. Routing at the intent level cannot separate "change my
   seat" from "how does standby work", and those are the same intent.

10. Some `auto` decisions are deflections in disguise. The agent sometimes routes `auto` and
    then writes a reply that asks the customer to DM. That case is not closed, a human still
    picks it up, and it is counted as coverage. The auto-handle rate is inflated by an amount
    I have not measured, and the fix is to check the drafted reply for a handoff before
    accepting an `auto` route.

11. The rubric punishes correctly saying nothing. One message aimed at @AmericanAir got an
    empty reply, which is the right call, and the judge scored it 1/1/1. "No reply needed" is
    a legitimate action the rubric has no way to express, so it is scored as a total failure.

12. `sendable` at 25.5% is not a deployable number. Three quarters of the agent's drafts are
    ones the judge would not post unedited. Read alongside the 2.7% false-auto rate, the
    honest summary is that the routing is close to trustworthy and the drafting is not. The
    system is a triage assistant that a human still edits, not an autoresponder.

## With another week

1. A second annotator on 100 items, for an inter-annotator ceiling. Without it you cannot tell
   a model at 80% from a task where humans agree 80% of the time.
2. Pick the escalation threshold from a cost model instead of hard-coding 0.60. Ask what an
   over-escalation costs relative to a false-auto, then choose the operating point on the
   coverage/risk curve.
3. Fix or drop retrieval. The ablation says it currently adds nothing, so either move to
   embeddings and re-run the same ablation, or cut it and simplify the system. Shipping a
   retrieval step that its own ablation cannot justify is the worst of the three options.
4. Adversarial judge validation: feed it replies with invented flight numbers and fabricated
   compensation and check it catches them. Agreeing with a human on ordinary replies does not
   prove it detects the failure it exists to detect.
5. Evaluate the escalation reason, not just the decision. Nothing currently checks whether the
   stated reason is true, and a plausible wrong reason is worse than none.
6. Combine the strata with a principled prevalence weight instead of reporting them side by side.

## Layout

```
cmd/
  build-dataset      twcs.csv -> per-brand episodes
  discover-intents   clustering + LLM naming -> taxonomy proposal
  sample-golden      two-stratum sampling
  label-assist       two-pass + adjudication drafts
  review             human console: labels | audit | replies
  run-agent          five systems over the golden set
  judge              LLM judge + blind human rating tasks
  report             metrics -> RESULTS.md + metrics.json
  gen-docs           regenerates the annotation guide
internal/
  data       episode extraction, cleaning, temporal split
  textx      tokenizer, TF-IDF, inverted index, k-means
  ml         logistic regression, stratified folds
  taxonomy   the 10 intents, the route policy
  agent      retriever, escalation policy, pipeline
  baseline   trivial and simple
  eval       metrics, CIs, judge, ordinal agreement
  llm        gemini | anthropic | cli providers, disk cache
```

[`reports/DECISIONS.md`](reports/DECISIONS.md) has the 17 non-obvious calls and the reasoning.

## LLM layer

Three providers behind one interface, chosen by `LLM_PROVIDER` or inferred from whichever key
is present. Every completion is cached to `cache/llm_cache.jsonl`, keyed on the request and not
the provider, so a cache built on one provider is reused by another. It is committed, which is
what makes `make repro` work offline and why a reviewer needs no key.

Gemini's default safety filters are off for this workload. Support tweets contain profanity and
descriptions of conflict, and filtering them biases every metric toward the polite half of the
corpus. Anything still blocked surfaces as an error rather than a silent drop.

## Borrowed

* Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
  Kaggle, `thoughtvector`. Pulled from the [SunidhiSriram/twcs](https://huggingface.co/datasets/SunidhiSriram/twcs)
  mirror, which hosts the identical csv, because Kaggle needs credentials.
* TF-IDF follows scikit-learn's `TfidfVectorizer` defaults, smoothed IDF `ln((1+n)/(1+df))+1`,
  sublinear TF, L2 norm, so the numbers compare to a standard implementation. Written from
  scratch in `internal/textx`, no code copied.
* k-means++ seeding: Arthur and Vassilvitskii, 2007.
* Wilson score interval: Wilson, 1927.
* Cohen's kappa, 1960, and quadratic-weighted kappa, 1968.
* Judge design (pointwise scoring to dodge position bias, judge larger than the system under
  test, blind to system identity) follows the MT-Bench / LLM-as-a-Judge line, Zheng et al., 2023.
* Claude Code was used as a coding assistant throughout. The taxonomy, routing policy and
  evaluation design are mine.
