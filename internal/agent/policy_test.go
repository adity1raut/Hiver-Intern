package agent

import (
	"strings"
	"testing"
)

func TestGuardrailsAreAOneWayRatchet(t *testing.T) {
	// The model wants to auto-handle a discrimination complaint. The policy must
	// override it. This is the single most important behaviour in the system.
	v := Decide("the gate agent was racist to my mother and nobody helped",
		Signals{Intent: "praise", IntentConfidence: 1.0, RetrievalTopScore: 1.0, ModelRoute: RouteAuto},
		DefaultThresholds)
	if v.Route != RouteEscalate {
		t.Fatalf("route = %v, want escalate", v.Route)
	}
	if !v.Overridden {
		t.Error("expected Overridden to record that the model said auto")
	}
	if !contains(v.FiredRules, "discrimination_or_conduct") {
		t.Errorf("fired rules = %v, want discrimination_or_conduct", v.FiredRules)
	}
}

func TestRulesNeverFlipEscalateToAuto(t *testing.T) {
	v := Decide("thanks for the great flight!",
		Signals{Intent: "praise", IntentConfidence: 1.0, RetrievalTopScore: 1.0,
			ModelRoute: RouteEscalate, ModelReason: "the model was unsure"},
		DefaultThresholds)
	if v.Route != RouteEscalate {
		t.Fatalf("a model escalation must stand, got %v", v.Route)
	}
}

func TestCleanPraiseCanAutoHandle(t *testing.T) {
	v := Decide("thanks for the smooth flight to ATL today, crew were lovely",
		Signals{Intent: "praise", IntentConfidence: 0.95, RetrievalTopScore: 0.4, ModelRoute: RouteAuto},
		DefaultThresholds)
	if v.Route != RouteAuto {
		t.Fatalf("route = %v (%s), want auto", v.Route, v.Reason)
	}
	if len(v.FiredRules) != 0 {
		t.Errorf("unexpected rules fired: %v", v.FiredRules)
	}
}

func TestLowConfidenceForcesEscalation(t *testing.T) {
	v := Decide("nice one",
		Signals{Intent: "praise", IntentConfidence: 0.2, RetrievalTopScore: 0.9, ModelRoute: RouteAuto},
		DefaultThresholds)
	if v.Route != RouteEscalate || !contains(v.FiredRules, "low_intent_confidence") {
		t.Fatalf("got %v %v", v.Route, v.FiredRules)
	}
}

func TestNoPrecedentForcesEscalation(t *testing.T) {
	v := Decide("nice one",
		Signals{Intent: "praise", IntentConfidence: 0.99, RetrievalTopScore: 0.01, ModelRoute: RouteAuto},
		DefaultThresholds)
	if v.Route != RouteEscalate || !contains(v.FiredRules, "no_precedent") {
		t.Fatalf("got %v %v", v.Route, v.FiredRules)
	}
}

func TestEscalationIntentsDefaultToHuman(t *testing.T) {
	v := Decide("I need to move my seat",
		Signals{Intent: "booking_change", IntentConfidence: 0.99, RetrievalTopScore: 0.9, ModelRoute: RouteAuto},
		DefaultThresholds)
	if v.Route != RouteEscalate || !contains(v.FiredRules, "intent_default_escalate") {
		t.Fatalf("got %v %v", v.Route, v.FiredRules)
	}
}

func TestMoneyAndPIILanguageEscalates(t *testing.T) {
	for _, msg := range []string{
		"I want a refund for this mess",
		"here is my confirmation number ABC123",
		"you charged me twice for the same bag",
		"I need to speak to a human",
		"my wheelchair was left behind",
		"my attorney will be in touch",
	} {
		v := Decide(msg, Signals{Intent: "praise", IntentConfidence: 1, RetrievalTopScore: 1, ModelRoute: RouteAuto}, DefaultThresholds)
		if v.Route != RouteEscalate {
			t.Errorf("%q was auto-handled (rules: %v)", msg, v.FiredRules)
		}
	}
}

func TestReasonIsAlwaysStated(t *testing.T) {
	for _, s := range []Signals{
		{Intent: "praise", IntentConfidence: 1, RetrievalTopScore: 1, ModelRoute: RouteAuto},
		{Intent: "baggage", IntentConfidence: 1, RetrievalTopScore: 1, ModelRoute: RouteAuto},
		{Intent: "praise", IntentConfidence: 0.1, RetrievalTopScore: 1, ModelRoute: RouteAuto},
	} {
		if v := Decide("hello there friends", s, DefaultThresholds); v.Reason == "" {
			t.Errorf("empty reason for %+v", s)
		}
	}
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// Regression tests for false positives found by smoke-testing the real corpus.
func TestGuardrailsDoNotFireOnInnocentPhrasing(t *testing.T) {
	cases := []struct{ msg, mustNotFire string }{
		// "the account you disabled" is not a disabled passenger
		{"why won't you call a Medallion member whose account you disabled", "vulnerable_passenger"},
		// "I press 1" is a phone menu, not the press
		{"I keep getting cut off after I press 1 on your phone menu", "media_or_virality"},
		// "hit me up" is not an assault report
		{"hit me up when the gate is announced", "safety_or_medical"},
	}
	for _, c := range cases {
		v := Decide(c.msg, Signals{Intent: "contact_support", IntentConfidence: 0.9,
			RetrievalTopScore: 0.5, ModelRoute: RouteAuto}, DefaultThresholds)
		if contains(v.FiredRules, c.mustNotFire) {
			t.Errorf("%q wrongly fired %s (all: %v)", c.msg, c.mustNotFire, v.FiredRules)
		}
	}
}

func TestGuardrailsStillFireOnTheRealThing(t *testing.T) {
	cases := []struct{ msg, mustFire string }{
		{"my disabled mother was left at the gate with no wheelchair", "vulnerable_passenger"},
		{"I am a reporter writing about this incident", "media_or_virality"},
		{"a passenger assaulted my son on flight 1422", "safety_or_medical"},
	}
	for _, c := range cases {
		v := Decide(c.msg, Signals{Intent: "praise", IntentConfidence: 0.9,
			RetrievalTopScore: 0.5, ModelRoute: RouteAuto}, DefaultThresholds)
		if !contains(v.FiredRules, c.mustFire) {
			t.Errorf("%q failed to fire %s (all: %v)", c.msg, c.mustFire, v.FiredRules)
		}
	}
}

// Every escalation reason is concatenated after "Escalated because it ...", so
// each fragment must read grammatically in that position.
func TestEscalationReasonsReadAsEnglish(t *testing.T) {
	seen := map[string]bool{}
	check := func(v Verdict) {
		if v.Route != RouteEscalate || seen[v.Reason] {
			return
		}
		seen[v.Reason] = true
		if strings.HasPrefix(v.Reason, "Escalated because it the ") ||
			strings.HasPrefix(v.Reason, "Escalated because it intent ") ||
			strings.HasPrefix(v.Reason, "Escalated because it no ") {
			t.Errorf("ungrammatical reason: %q", v.Reason)
		}
	}
	for _, msg := range []string{
		"I want a refund", "let me speak to a human", "my attorney will call",
		"the crew was racist", "here is my confirmation number ABC123", "nice flight",
	} {
		check(Decide(msg, Signals{Intent: "praise", IntentConfidence: 0.95,
			RetrievalTopScore: 0.5, ModelRoute: RouteAuto}, DefaultThresholds))
	}
	check(Decide("hello", Signals{Intent: "praise", IntentConfidence: 0.1,
		RetrievalTopScore: 0.5, ModelRoute: RouteAuto}, DefaultThresholds))
	check(Decide("hello", Signals{Intent: "praise", IntentConfidence: 0.95,
		RetrievalTopScore: 0.0, ModelRoute: RouteAuto}, DefaultThresholds))
	check(Decide("hello", Signals{Intent: "baggage", IntentConfidence: 0.95,
		RetrievalTopScore: 0.5, ModelRoute: RouteAuto}, DefaultThresholds))
}
