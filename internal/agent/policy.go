package agent

import (
	"regexp"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
)

// Route is the handling decision for a message.
type Route string

const (
	RouteAuto     Route = "auto"
	RouteEscalate Route = "escalate"
)

// Rule is a deterministic escalation guardrail. Rules only ever push towards
// escalation, never away from it: a desk survives an agent that over-escalates,
// not one that auto-replies to a discrimination complaint.
type Rule struct {
	Name   string
	Reason string
	re     *regexp.Regexp
}

func rule(name, reason, pattern string) Rule {
	return Rule{Name: name, Reason: reason, re: regexp.MustCompile(`(?i)` + pattern)}
}

// HardRules match against the raw customer message.
var HardRules = []Rule{
	rule("safety_or_medical",
		"mentions a safety, medical or emergency situation that a human must own",
		`\b(emergency|ambulance|paramedic|medical|seizure|heart attack|died|passed away|death|funeral|hospital|injur(y|ed)|assault(ed)?|threat(en(ed|ing))?|unsafe|evacuat|smoke in the cabin|turbulence injur)\b`),
	rule("legal_or_regulatory",
		"raises legal or regulatory exposure",
		`\b(lawyer|attorney|sue|suing|lawsuit|legal action|small claims|DOT complaint|department of transportation|FAA|litigation|subpoena)\b`),
	rule("discrimination_or_conduct",
		"alleges discrimination or serious staff misconduct",
		`\b(racist|racism|discriminat(e|ed|ion|ory)|sexist|homophob|profil(ed|ing)|harass(ed|ment)?|kicked (me|us) off|removed from the (flight|plane)|police|air marshal)\b`),
	// "disabled" alone matched "the account you disabled"; it must carry passenger context.
	rule("vulnerable_passenger",
		"involves a passenger needing special assistance",
		`\b(wheelchair|disabilit(y|ies)|disabled (passenger|person|traveller|traveler|child|veteran|mother|father)|service (dog|animal)|unaccompanied minor|my (baby|infant|toddler)|special assistance|oxygen tank|autis)\b`),
	rule("money_claim",
		"is a claim on money or miles and needs an agent with account access",
		`\b(refund|reimburse|compensat|voucher|charged? me|double charg|overcharg|chargeback|money back|credit back|my miles (are )?(gone|missing)|stolen)\b`),
	rule("account_or_pii",
		"requires account or booking data the agent must not handle in public",
		`\b(confirmation (number|code)|record locator|booking (ref|reference|number)|ticket number|credit card|card number|passport|skymiles (number|account)|last four|SSN)\b`),
	rule("human_requested",
		"contains an explicit request for a human",
		`\b(speak to (a|an) (human|person|manager|supervisor|agent)|real person|talk to someone|get me a manager|escalate)\b`),
	// bare "press" matched "I press 1" in a phone-menu complaint.
	rule("media_or_virality",
		"is a press or public-escalation risk",
		`\b(journalist|reporter|the press|press coverage|news media|going viral|local news|@?FoxNews|@?CNN|blog(ging)? about|filing a report with)\b`),
}

// Signals are the non-textual inputs to the routing decision.
type Signals struct {
	Intent            string
	IntentConfidence  float64
	RetrievalTopScore float64
	ModelRoute        Route
	ModelReason       string
}

// Thresholds tune the confidence-based guardrails.
type Thresholds struct {
	MinIntentConfidence float64 `yaml:"min_intent_confidence"`
	MinRetrievalScore   float64 `yaml:"min_retrieval_score"`
}

// DefaultThresholds are the shipped values.
var DefaultThresholds = Thresholds{MinIntentConfidence: 0.60, MinRetrievalScore: 0.12}

// Verdict is the routing decision and its stated cause.
type Verdict struct {
	Route      Route    `json:"route"`
	Reason     string   `json:"reason"`
	FiredRules []string `json:"fired_rules,omitempty"`
	Overridden bool     `json:"overridden"` // a rule flipped the model's auto to escalate
}

// Decide combines the model's proposed route with the guardrails.
func Decide(msg string, s Signals, th Thresholds) Verdict {
	var fired []string
	var reasons []string
	for _, r := range HardRules {
		if r.re.MatchString(msg) {
			fired = append(fired, r.Name)
			reasons = append(reasons, r.Reason)
		}
	}
	if s.IntentConfidence < th.MinIntentConfidence {
		fired = append(fired, "low_intent_confidence")
		reasons = append(reasons, "was classified with too little confidence to act on")
	}
	if s.RetrievalTopScore < th.MinRetrievalScore {
		fired = append(fired, "no_precedent")
		reasons = append(reasons, "has no sufficiently similar precedent in the brand's history to ground a reply")
	}
	if in, ok := taxonomy.Get(s.Intent); ok && in.DefaultAction == "escalate" {
		fired = append(fired, "intent_default_escalate")
		reasons = append(reasons, "falls under intent "+s.Intent+", which a human queue owns by default")
	}

	if len(fired) > 0 {
		return Verdict{
			Route:      RouteEscalate,
			Reason:     "Escalated because it " + joinReasons(reasons) + ".",
			FiredRules: fired,
			Overridden: s.ModelRoute == RouteAuto,
		}
	}
	if s.ModelRoute == RouteEscalate {
		reason := strings.TrimSpace(s.ModelReason)
		if reason == "" {
			reason = "the model judged this message to need a human."
		}
		return Verdict{Route: RouteEscalate, Reason: reason, FiredRules: []string{"model_judgement"}}
	}
	return Verdict{
		Route:  RouteAuto,
		Reason: "Auto-handled: intent " + s.Intent + " is self-serviceable, no guardrail fired, and a close precedent exists.",
	}
}

func joinReasons(rs []string) string {
	switch len(rs) {
	case 0:
		return ""
	case 1:
		return rs[0]
	case 2:
		return rs[0] + " and " + rs[1]
	default:
		return strings.Join(rs[:len(rs)-1], ", ") + ", and " + rs[len(rs)-1]
	}
}

// Matches reports whether the rule fires on a message.
func (r Rule) Matches(s string) bool { return r.re.MatchString(s) }
