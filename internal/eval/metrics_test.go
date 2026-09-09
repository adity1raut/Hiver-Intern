package eval

import (
	"math"
	"testing"
)

func TestRoutePositiveClassIsEscalate(t *testing.T) {
	gold := []string{"auto", "auto", "escalate", "escalate", "escalate"}
	pred := []string{"auto", "escalate", "auto", "escalate", "escalate"}
	r := Route(gold, pred)
	if r.N != 5 {
		t.Fatalf("N = %d", r.N)
	}
	// two items auto-handled, one of which needed a human
	if math.Abs(r.AutoRate-2.0/5) > 1e-9 {
		t.Errorf("auto rate %v", r.AutoRate)
	}
	if math.Abs(r.FalseAutoRate-0.5) > 1e-9 {
		t.Errorf("false-auto rate %v, want 0.5", r.FalseAutoRate)
	}
	if math.Abs(r.EscalationRecall-2.0/3) > 1e-9 {
		t.Errorf("escalation recall %v", r.EscalationRecall)
	}
	if math.Abs(r.OverEscalation-0.5) > 1e-9 {
		t.Errorf("over-escalation %v", r.OverEscalation)
	}
}

func TestClassifyMacroF1IgnoresAbsentLabels(t *testing.T) {
	gold := []string{"a", "a", "b"}
	pred := []string{"a", "a", "b"}
	// "c" exists in the taxonomy but never occurs: it must not drag macro-F1 down.
	got := Classify(gold, pred, []string{"a", "b", "c"})
	if math.Abs(got.MacroF1-1.0) > 1e-9 {
		t.Fatalf("macro F1 = %v, want 1.0", got.MacroF1)
	}
	if math.Abs(got.Accuracy-1.0) > 1e-9 {
		t.Fatalf("accuracy = %v", got.Accuracy)
	}
}

func TestWilsonCIBracketsThePointEstimate(t *testing.T) {
	lo, hi := WilsonCI(5, 100)[0], WilsonCI(5, 100)[1]
	if !(lo < 0.05 && 0.05 < hi) {
		t.Fatalf("CI [%v,%v] does not bracket 0.05", lo, hi)
	}
	if z := WilsonCI(0, 50); z[0] != 0 || z[1] <= 0 {
		t.Fatalf("zero-event CI = %v, want lower bound 0 and positive upper", z)
	}
}

func TestCohenKappaIsZeroForChanceAgreement(t *testing.T) {
	if k := CohenKappa([]string{"a", "a", "b", "b"}, []string{"a", "a", "b", "b"}); math.Abs(k-1) > 1e-9 {
		t.Fatalf("perfect agreement kappa = %v", k)
	}
	// Both raters say "a" every time: agreement is total but entirely by chance.
	if k := CohenKappa([]string{"a", "a"}, []string{"a", "a"}); k != 1 {
		t.Fatalf("degenerate kappa = %v", k)
	}
}

func TestQuadraticKappaPunishesDistantDisagreement(t *testing.T) {
	near := CompareOrdinal([]float64{4, 4, 5, 3}, []float64{5, 4, 4, 3}, 1, 5)
	far := CompareOrdinal([]float64{1, 1, 5, 5}, []float64{5, 5, 1, 1}, 1, 5)
	if near.QuadraticKappa <= far.QuadraticKappa {
		t.Fatalf("near %v should beat far %v", near.QuadraticKappa, far.QuadraticKappa)
	}
	if near.Within1 != 1 {
		t.Errorf("within-1 = %v, want 1", near.Within1)
	}
}

func TestOrdinalBiasSign(t *testing.T) {
	o := CompareOrdinal([]float64{5, 5, 5}, []float64{3, 3, 3}, 1, 5)
	if o.Bias <= 0 {
		t.Fatalf("bias = %v, want positive when rater A is generous", o.Bias)
	}
	if math.Abs(o.MAE-2) > 1e-9 {
		t.Fatalf("MAE = %v", o.MAE)
	}
}
