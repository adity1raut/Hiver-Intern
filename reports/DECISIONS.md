# Decision log

The non-obvious calls, and why. Numbers referenced here are reproduced by `make repro`.

---

**1. Brand: Delta, chosen on reply substance rather than volume.**
AmazonHelp (169k replies) and AppleSupport (107k) are far larger, but I measured the
share of each brand's first replies that are a bare "DM us"/link deflection: Delta 23%,
AmericanAir 25%, AmazonHelp 45%, SpotifyCares 59%, AppleSupport 77%, TMobileHelp 90%.
A brand that deflects 90% of the time has no in-channel resolution behaviour to learn
or to be graded against — the reply task would collapse into predicting one canned line.
Delta had the lowest deflection rate of any high-volume brand plus 42k replies, so it
has both scale and something to imitate.

**2. The unit of work is the first customer message of a thread, not every tweet.**
An "episode" is a thread rooted at an inbound tweet where Delta replied. This is what a
triage agent actually sees: a cold public message with no context. Scoring mid-thread
turns would leak the resolution path into the input and inflate everything.

**3. Temporal split, not random.** History (earliest 70%, 17,979 episodes) is the only
thing retrieval may index; the eval pool is the latest 30%. Delta's volume is concentrated
in Oct–Dec 2017 and mass-disruption days generate near-identical tweets minutes apart. Under
a random split the retriever would routinely find a *twin* of the test message — same
incident, same hour — and the reply score would measure duplicate detection, not generalisation.

**4. Intents are defined by what the desk does next, not by topic.** Clustering produced
30 groups, seven of which were near-duplicate flavours of "positive feedback". Merging on
topic similarity would have kept them apart and split things that route identically. The
shipped 10 intents are the coarsest partition where each cell has a distinct next action
and a distinct information requirement. `refund_compensation` is split out from
`service_complaint` for exactly this reason: both are angry customers, but only one is a
claim on money.

**5. `praise` is a first-class intent, not noise.** It is one of the largest slices of
Delta's inbound. Excluding it would have removed most of the traffic the agent can safely
auto-handle, which is precisely where the business value is.

**6. Escalation guardrails are a one-way ratchet.** The LLM proposes a route; deterministic
regex rules can push it towards `escalate` and never away. A desk survives an agent that
hands too much to a human; it does not survive one that auto-replies to a discrimination
complaint. This asymmetry is enforced in code (`internal/agent/policy.go`) and asserted in
tests, not left to the prompt.

**7. Two of the guardrails are confidence-based, and I report their calibration anyway.**
`low_intent_confidence` and `no_precedent` gate on numbers the system produces about itself.
Self-reported LLM confidence is usually badly calibrated, so §8 of RESULTS.md measures it
directly rather than assuming the threshold does anything. If the reliability curve is flat,
that threshold is decoration and the report says so.

**8. The trivial baseline is "always DM us", which is genuinely what Delta does 23% of
the time.** A straw-man baseline proves nothing. This one is a real competitor on
reply-similarity metrics precisely because it mimics Delta's most common behaviour, which
is what makes it a useful demonstration of how little such metrics measure.

**9. Two trivial baselines, not one.** `trivial-escalate` has a 0% false-auto rate and 0%
coverage; `trivial-auto` has maximum coverage and a catastrophic false-auto rate. Together
they bound the tradeoff, so no single number can be quoted without its cost.

**10. The simple baseline copies real Delta replies verbatim.** Nearest-neighbour retrieval
plus a literal copy cannot hallucinate — every word was written by a Delta agent. That makes
it very hard to beat on grounding, and it isolates what the LLM actually adds: relevance when
no close precedent exists. Its classifier is trained under 5-fold CV *on the golden set*,
which means the baseline sees labelled training data the zero-shot agent never does. That
handicaps the agent deliberately.

**11. The judge grades against a rubric, not against Delta's actual reply.** It is shown
Delta's reply as *one acceptable answer* and told explicitly that Delta's own replies are
often lazy deflections that would themselves score poorly. Treating the historical reply as
ground truth would have made "please DM us" the optimal output and punished any reply that
was better than what Delta sent.

**12. Delta's own replies are graded by the same judge, as a reference row.** Comparing the
agent to a baseline says whether it is better than nothing. Comparing it to the human desk
says whether it is good enough to deploy. The second is the question that matters, and it is
the only row in the table that a reviewer can sanity-check against their own intuition.

**13. The headline reply-quality number is `auto-sendable`, not `sendable`.** Overall
sendable rate averages over replies the system would never have sent unattended, which
flatters any system that escalates a lot. Restricting to replies actually routed `auto` is
the only version that describes what a customer would really receive.

**14. Judge and agent are different models.** The judge runs on the stronger model, the agent
on the cheaper one. A judge that shares the agent's weights shares its blind spots and rates
its own output generously; the size gap is a cheap partial defence against self-preference bias.

**15. Human review is split into anchored and blind halves.** Reviewing all 200 items with the
draft label visible is fast but anchored — the override rate it produces is a lower bound on
true disagreement. A separate random subset is re-labelled cold. Only the blind number is
quoted as evidence of label quality, and the report says so explicitly.

**16. The LLM cache is committed to the repo and keyed on request content, not on provider.**
This is what makes `make repro` run offline in minutes with no API key, and it means a cache
built on one provider is reused by another. `LLM_PROVIDER=offline` turns any cache miss into a
hard error, so the repro target either reproduces the committed numbers exactly or fails loudly
rather than silently re-generating different ones.

**17. Zero third-party Go dependencies.** TF-IDF, the inverted index, spherical k-means,
multinomial logistic regression, Wilson intervals, bootstrap CIs and quadratic-weighted kappa
are all implemented in `internal/`. This is ~700 lines I would normally not write, but it makes
`go build ./...` hermetic, keeps the reproduce step to a couple of minutes, and means every
metric definition is inspectable rather than hidden behind a library default.
