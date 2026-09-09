package agent

import "testing"

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
