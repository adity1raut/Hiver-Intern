// Package baseline implements the two reference systems the agent must beat.
package baseline

import (
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/agent"
	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/ml"
	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

// ---------------------------------------------------------------- trivial

// Trivial is the floor: one constant intent, one canned reply, one fixed route.
// The canned reply is not a straw man - roughly a quarter of Delta's real replies
// are a variant of "please DM us", so it is a genuine competitor on any
// reply-similarity metric.
type Trivial struct {
	Intent string // majority class
	Route  agent.Route
	Reply  string
	Name   string
}

const CannedReply = "I'm sorry for the trouble. Please send us a DM with your confirmation number so we can take a look. *AI"

func (t Trivial) Run(episodeID, msg string) agent.Output {
	return agent.Output{
		EpisodeID: episodeID, CustomerText: msg, System: t.Name,
		Intent: t.Intent, Confidence: 1.0, Reply: t.Reply,
		Route:       t.Route,
		RouteReason: "Fixed policy: this system always routes to " + string(t.Route) + ".",
	}
}

// ---------------------------------------------------------------- simple

// Simple is a competent no-LLM system: cross-validated TF-IDF logistic regression
// for intent, a verbatim copy of the nearest historical reply, and the guardrails
// for routing. It cannot hallucinate - every word was written by a Delta agent -
// but it cannot be relevant when no close precedent exists.
type Simple struct {
	Clf       *ml.LogReg
	Vec       *textx.Vectorizer
	Retriever *agent.Retriever
	Th        agent.Thresholds
	Name      string
}

func (s Simple) Run(episodeID, msg string) agent.Output {
	out := agent.Output{EpisodeID: episodeID, CustomerText: msg, System: s.Name}

	intent, conf := s.Clf.Predict(s.Vec.Transform(msg))
	out.Intent, out.Confidence = intent, conf

	prec := s.Retriever.Retrieve(msg, 1)
	out.Precedents = prec
	if len(prec) > 0 {
		out.Reply = prec[0].BrandReply
		out.RetrievalTop = prec[0].Score
	} else {
		out.Reply = CannedReply
	}

	v := agent.Decide(msg, agent.Signals{
		Intent: intent, IntentConfidence: conf,
		RetrievalTopScore: out.RetrievalTop,
		ModelRoute:        agent.RouteAuto, // no model: rules decide alone
	}, s.Th)
	out.Route, out.RouteReason, out.FiredRules = v.Route, v.Reason, v.FiredRules
	return out
}

// MajorityIntent returns the most frequent label.
func MajorityIntent(labels []string) string {
	c := map[string]int{}
	for _, l := range labels {
		c[l]++
	}
	best, n := "other", -1
	for l, k := range c {
		if k > n || (k == n && l < best) {
			best, n = l, k
		}
	}
	return best
}

// ReplyFromThread returns the brand's actual reply.
func ReplyFromThread(e data.Episode) string { return strings.TrimSpace(e.BrandReply) }
