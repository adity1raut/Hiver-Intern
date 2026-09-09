package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/llm"
	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
)

// Output is everything the agent decided about one incoming message.
type Output struct {
	EpisodeID    string      `json:"episode_id"`
	CustomerText string      `json:"customer_text"`
	System       string      `json:"system"`
	Intent       string      `json:"intent"`
	Confidence   float64     `json:"confidence"`
	Reply        string      `json:"reply"`
	Route        Route       `json:"route"`
	RouteReason  string      `json:"route_reason"`
	FiredRules   []string    `json:"fired_rules,omitempty"`
	Overridden   bool        `json:"overridden"`
	ModelRoute   Route       `json:"model_route"`
	ModelReason  string      `json:"model_reason"`
	RetrievalTop float64     `json:"retrieval_top"`
	Precedents   []Precedent `json:"precedents"`
	Error        string      `json:"error,omitempty"`
}

// Agent is the full system: retrieve precedents, classify + draft + propose a
// route in one grounded LLM call, then apply the deterministic guardrails.
type Agent struct {
	Client     llm.Client
	Retriever  *Retriever
	Model      string
	K          int
	Thresholds Thresholds
	// NoRetrieval runs the ablation that drops precedents from the prompt.
	NoRetrieval bool
	Name        string
}

const systemPrompt = `You are the Twitter/X support agent for Delta Air Lines (@Delta).

You do three things for each incoming public tweet:
1. Classify it into exactly one intent from the taxonomy.
2. Draft the public reply Delta would send, imitating how Delta actually replied to similar cases.
3. Say whether the message can be auto-handled or needs a human, and why.

House style, learned from Delta's real replies:
- One or two sentences. Public tweets are short.
- Open by acknowledging the specific thing that happened, not a generic apology.
- Never invent a fact: no flight numbers, times, gate numbers, refund amounts, policy
  figures or compensation promises that are not in the customer's message.
- Anything needing account or booking data moves to DM - say so rather than asking for
  a confirmation number in public.
- Sign off with an agent-initials token in the form *AB. Use *AI.
- No hashtags. No emoji unless the customer's message is celebratory.

Reply with ONLY a JSON object.`

// Run processes one message.
func (a *Agent) Run(ctx context.Context, episodeID, msg string) Output {
	out := Output{EpisodeID: episodeID, CustomerText: msg, System: a.Name}

	var precedents []Precedent
	if !a.NoRetrieval {
		precedents = a.Retriever.Retrieve(msg, a.K)
		out.RetrievalTop = a.Retriever.TopScore(msg)
	} else {
		// The ablation still gets the retrieval score so the policy is comparable;
		// only the *prompt* loses the precedents.
		out.RetrievalTop = a.Retriever.TopScore(msg)
	}
	out.Precedents = precedents

	var resp struct {
		Intent      string  `json:"intent"`
		Confidence  float64 `json:"confidence"`
		Reply       string  `json:"reply"`
		Route       string  `json:"route"`
		RouteReason string  `json:"route_reason"`
	}
	err := llm.CompleteJSON(ctx, a.Client, llm.Request{
		Model: a.Model, MaxTokens: 700, Temperature: 0,
		System: systemPrompt,
		Prompt: a.buildPrompt(msg, precedents),
	}, &resp)
	if err != nil {
		out.Error = err.Error()
		out.Intent = "other"
		out.Route, out.RouteReason = RouteEscalate, "Escalated because the agent failed to produce a decision: "+err.Error()
		out.FiredRules = []string{"agent_error"}
		return out
	}

	out.Intent = taxonomy.Normalise(resp.Intent)
	out.Confidence = clamp01(resp.Confidence)
	out.Reply = strings.TrimSpace(resp.Reply)
	out.ModelRoute = normaliseRoute(resp.Route)
	out.ModelReason = strings.TrimSpace(resp.RouteReason)

	v := Decide(msg, Signals{
		Intent:            out.Intent,
		IntentConfidence:  out.Confidence,
		RetrievalTopScore: out.RetrievalTop,
		ModelRoute:        out.ModelRoute,
		ModelReason:       out.ModelReason,
	}, a.Thresholds)

	out.Route, out.RouteReason = v.Route, v.Reason
	out.FiredRules, out.Overridden = v.FiredRules, v.Overridden
	return out
}

func (a *Agent) buildPrompt(msg string, precedents []Precedent) string {
	var b strings.Builder
	b.WriteString("INTENT TAXONOMY\n")
	b.WriteString(taxonomy.PromptBlock())

	if len(precedents) > 0 {
		b.WriteString("\nHOW DELTA HANDLED SIMILAR MESSAGES BEFORE\n")
		b.WriteString("(retrieved from Delta's own reply history; use these for tone, structure and what Delta\n")
		b.WriteString("does and does not promise. Do NOT copy facts specific to those customers.)\n\n")
		for i, p := range precedents {
			fmt.Fprintf(&b, "[%d] (similarity %.2f)\n  customer: %s\n  Delta:    %s\n", i+1, p.Score, p.Customer, p.BrandReply)
		}
	} else {
		b.WriteString("\n(No historical precedents are available for this message.)\n")
	}

	b.WriteString("\nINCOMING MESSAGE\n")
	b.WriteString(msg)

	b.WriteString(`

Return this JSON:
{
  "intent": "<one intent name from the taxonomy>",
  "confidence": <0.0-1.0, how sure you are of the intent>,
  "reply": "<the public reply Delta should send>",
  "route": "<auto|escalate>",
  "route_reason": "<one sentence: why this can or cannot be sent without a human>"
}

Choose "escalate" when a human must see this before or instead of your reply: the
customer needs account data, is owed money, is angry enough that a canned reply makes
it worse, raises a safety/legal/discrimination issue, or the case is ambiguous. Choose
"auto" only when your drafted reply fully and safely serves the customer on its own.`)
	return b.String()
}

func normaliseRoute(s string) Route {
	s = strings.ToLower(strings.TrimSpace(s))
	if strings.HasPrefix(s, "auto") {
		return RouteAuto
	}
	return RouteEscalate
}

func clamp01(f float64) float64 {
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}
