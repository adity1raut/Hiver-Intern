package ml

import (
	"testing"

	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

func TestLogRegSeparatesTwoTopics(t *testing.T) {
	docs := []string{
		"my bag is lost", "lost luggage help", "bag never arrived", "missing suitcase",
		"flight delayed three hours", "my flight is cancelled", "delayed again", "cancelled flight",
	}
	y := []int{0, 0, 0, 0, 1, 1, 1, 1}
	var v textx.Vectorizer
	v.Fit(docs, 1, 0)
	m := &LogReg{Epochs: 400, LR: 0.5, Seed: 1}
	m.Fit(v.TransformAll(docs), y, v.Dim(), []string{"baggage", "disruption"})

	if got, p := m.Predict(v.Transform("lost bag")); got != "baggage" || p < 0.5 {
		t.Errorf("got %s (%.2f), want baggage", got, p)
	}
	if got, _ := m.Predict(v.Transform("flight cancelled")); got != "disruption" {
		t.Errorf("got %s, want disruption", got)
	}
}

func TestPredictProbaSumsToOne(t *testing.T) {
	var v textx.Vectorizer
	v.Fit([]string{"a b", "c d"}, 1, 0)
	m := &LogReg{Epochs: 10, Seed: 1}
	m.Fit(v.TransformAll([]string{"a b", "c d"}), []int{0, 1}, v.Dim(), []string{"x", "y"})
	p := m.PredictProba(v.Transform("a b"))
	sum := 0.0
	for _, x := range p {
		sum += x
	}
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("probabilities sum to %v", sum)
	}
}

func TestStratifiedFoldsKeepClassBalance(t *testing.T) {
	y := make([]int, 100)
	for i := range y {
		y[i] = i % 4
	}
	fold := StratifiedFolds(y, 5, 1)
	counts := map[[2]int]int{}
	for i, f := range fold {
		counts[[2]int{f, y[i]}]++
	}
	for f := 0; f < 5; f++ {
		for c := 0; c < 4; c++ {
			if n := counts[[2]int{f, c}]; n != 5 {
				t.Errorf("fold %d class %d has %d items, want 5", f, c, n)
			}
		}
	}
}
