// Package eval holds the metrics, the LLM judge and the judge-vs-human agreement
// analysis.
package eval

import (
	"math"
	"math/rand"
	"sort"
)

// ClassReport is precision/recall/F1 for one label.
type ClassReport struct {
	Label     string  `json:"label"`
	Support   int     `json:"support"`
	Predicted int     `json:"predicted"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// Classification is the full multi-class report.
type Classification struct {
	N         int           `json:"n"`
	Accuracy  float64       `json:"accuracy"`
	MacroF1   float64       `json:"macro_f1"`
	Weighted  float64       `json:"weighted_f1"`
	PerClass  []ClassReport `json:"per_class"`
	Labels    []string      `json:"labels"`
	Confusion [][]int       `json:"confusion"` // [gold][pred]
}

// Classify computes the multi-class report. labels fixes row/column order.
func Classify(gold, pred []string, labels []string) Classification {
	ix := map[string]int{}
	for i, l := range labels {
		ix[l] = i
	}
	k := len(labels)
	cm := make([][]int, k)
	for i := range cm {
		cm[i] = make([]int, k)
	}
	correct := 0
	n := 0
	for i := range gold {
		g, ok1 := ix[gold[i]]
		p, ok2 := ix[pred[i]]
		if !ok1 || !ok2 {
			continue
		}
		cm[g][p]++
		n++
		if g == p {
			correct++
		}
	}
	rep := Classification{N: n, Labels: labels, Confusion: cm}
	if n > 0 {
		rep.Accuracy = float64(correct) / float64(n)
	}
	var sumF1, wSum float64
	for c := 0; c < k; c++ {
		tp := cm[c][c]
		var fp, fn int
		for r := 0; r < k; r++ {
			if r != c {
				fp += cm[r][c]
				fn += cm[c][r]
			}
		}
		support := tp + fn
		predicted := tp + fp
		var p, r, f float64
		if predicted > 0 {
			p = float64(tp) / float64(predicted)
		}
		if support > 0 {
			r = float64(tp) / float64(support)
		}
		if p+r > 0 {
			f = 2 * p * r / (p + r)
		}
		rep.PerClass = append(rep.PerClass, ClassReport{labels[c], support, predicted, p, r, f})
		// Average only over labels present in the gold set; absent ones would
		// silently drag macro-F1 toward zero.
		if support > 0 {
			sumF1 += f
			wSum += float64(support)
			rep.Weighted += f * float64(support)
		}
	}
	present := 0
	for _, c := range rep.PerClass {
		if c.Support > 0 {
			present++
		}
	}
	if present > 0 {
		rep.MacroF1 = sumF1 / float64(present)
	}
	if wSum > 0 {
		rep.Weighted /= wSum
	}
	return rep
}

// Routing is the auto/escalate report. "escalate" is the positive class: the
// expensive error is auto-handling something that needed a human.
type Routing struct {
	N        int `json:"n"`
	AutoAuto int `json:"auto_auto"`     // coverage
	AutoEsc  int `json:"auto_escalate"` // wasted human time
	EscAuto  int `json:"escalate_auto"` // the dangerous cell
	EscEsc   int `json:"escalate_escalate"`

	AutoRate         float64    `json:"auto_rate"`            // share handled without a human
	FalseAutoRate    float64    `json:"false_auto_rate"`      // P(needed a human | auto-handled)
	EscalationRecall float64    `json:"escalation_recall"`    // P(escalated | needed a human)
	AutoPrecision    float64    `json:"auto_precision"`       // 1 - FalseAutoRate
	OverEscalation   float64    `json:"over_escalation_rate"` // P(escalated | did not need one)
	Accuracy         float64    `json:"accuracy"`
	FalseAutoCI      [2]float64 `json:"false_auto_ci95"`
	AutoRateCI       [2]float64 `json:"auto_rate_ci95"`
}

// Route computes routing metrics from "auto"/"escalate" label slices.
func Route(gold, pred []string) Routing {
	var r Routing
	for i := range gold {
		g, p := gold[i], pred[i]
		switch {
		case g == "auto" && p == "auto":
			r.AutoAuto++
		case g == "auto" && p == "escalate":
			r.AutoEsc++
		case g == "escalate" && p == "auto":
			r.EscAuto++
		case g == "escalate" && p == "escalate":
			r.EscEsc++
		default:
			continue
		}
		r.N++
	}
	auto := r.AutoAuto + r.EscAuto
	esc := r.EscAuto + r.EscEsc
	goldAuto := r.AutoAuto + r.AutoEsc
	if r.N > 0 {
		r.AutoRate = float64(auto) / float64(r.N)
		r.Accuracy = float64(r.AutoAuto+r.EscEsc) / float64(r.N)
	}
	if auto > 0 {
		r.FalseAutoRate = float64(r.EscAuto) / float64(auto)
		r.AutoPrecision = 1 - r.FalseAutoRate
	}
	if esc > 0 {
		r.EscalationRecall = float64(r.EscEsc) / float64(esc)
	}
	if goldAuto > 0 {
		r.OverEscalation = float64(r.AutoEsc) / float64(goldAuto)
	}
	r.FalseAutoCI = WilsonCI(r.EscAuto, auto)
	r.AutoRateCI = WilsonCI(auto, r.N)
	return r
}

// WilsonCI is the 95% Wilson score interval. At n=200 with rare events the normal
// approximation is badly wrong.
func WilsonCI(k, n int) [2]float64 {
	if n == 0 {
		return [2]float64{0, 0}
	}
	const z = 1.959964
	p := float64(k) / float64(n)
	den := 1 + z*z/float64(n)
	centre := (p + z*z/(2*float64(n))) / den
	half := z * math.Sqrt(p*(1-p)/float64(n)+z*z/(4*float64(n)*float64(n))) / den
	return [2]float64{math.Max(0, centre-half), math.Min(1, centre+half)}
}

// BootstrapCI resamples with replacement for a 95% interval on any statistic.
func BootstrapCI(n int, iters int, seed int64, stat func(idx []int) float64) [2]float64 {
	if n == 0 {
		return [2]float64{0, 0}
	}
	rng := rand.New(rand.NewSource(seed))
	vals := make([]float64, iters)
	idx := make([]int, n)
	for b := 0; b < iters; b++ {
		for i := range idx {
			idx[i] = rng.Intn(n)
		}
		vals[b] = stat(idx)
	}
	sort.Float64s(vals)
	lo := vals[int(0.025*float64(iters))]
	hi := vals[int(0.975*float64(iters))-1+1]
	if hi > vals[iters-1] {
		hi = vals[iters-1]
	}
	return [2]float64{lo, hi}
}

// CohenKappa is chance-corrected agreement over nominal labels.
func CohenKappa(a, b []string) float64 {
	if len(a) == 0 {
		return 0
	}
	labels := map[string]bool{}
	for _, x := range a {
		labels[x] = true
	}
	for _, x := range b {
		labels[x] = true
	}
	n := float64(len(a))
	agree := 0.0
	ca, cb := map[string]float64{}, map[string]float64{}
	for i := range a {
		if a[i] == b[i] {
			agree++
		}
		ca[a[i]]++
		cb[b[i]]++
	}
	po := agree / n
	pe := 0.0
	for l := range labels {
		pe += (ca[l] / n) * (cb[l] / n)
	}
	if pe == 1 {
		return 1
	}
	return (po - pe) / (1 - pe)
}

// Agreement is raw percent agreement.
func Agreement(a, b []string) float64 {
	if len(a) == 0 {
		return 0
	}
	n := 0
	for i := range a {
		if a[i] == b[i] {
			n++
		}
	}
	return float64(n) / float64(len(a))
}
