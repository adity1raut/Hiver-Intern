// Package ml holds the trained models the evaluation needs.
package ml

import (
	"math"
	"math/rand"
	"sort"

	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

// LogReg is multinomial logistic regression trained with SGD and L2 over sparse
// TF-IDF features. It backs the "simple" baseline: a real trained model, because a
// baseline the LLM beats only for being weak proves nothing.
type LogReg struct {
	Classes []string
	W       [][]float64 // [class][feature]
	B       []float64
	Dim     int

	LR     float64
	L2     float64
	Epochs int
	Seed   int64
}

// Fit trains on sparse vectors with integer labels indexing classes.
func (m *LogReg) Fit(X []textx.Sparse, y []int, dim int, classes []string) {
	if m.LR == 0 {
		m.LR = 0.5
	}
	if m.Epochs == 0 {
		m.Epochs = 300
	}
	if m.L2 == 0 {
		m.L2 = 1e-4
	}
	m.Classes, m.Dim = classes, dim
	k := len(classes)
	m.W = make([][]float64, k)
	for i := range m.W {
		m.W[i] = make([]float64, dim)
	}
	m.B = make([]float64, k)

	rng := rand.New(rand.NewSource(m.Seed))
	idx := make([]int, len(X))
	for i := range idx {
		idx[i] = i
	}
	probs := make([]float64, k)

	for ep := 0; ep < m.Epochs; ep++ {
		rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
		lr := m.LR / (1 + 0.01*float64(ep)) // simple decay
		for _, i := range idx {
			m.softmax(X[i], probs)
			for c := 0; c < k; c++ {
				g := probs[c]
				if c == y[i] {
					g -= 1
				}
				if g == 0 {
					continue
				}
				step := lr * g
				for kk, f := range X[i].Idx {
					m.W[c][f] -= step * float64(X[i].Val[kk])
				}
				m.B[c] -= step
			}
		}
		// L2 shrinkage once per epoch: cheaper than per-sample, same effect.
		shrink := 1 - lr*m.L2
		for c := range m.W {
			for f := range m.W[c] {
				m.W[c][f] *= shrink
			}
		}
	}
}

func (m *LogReg) softmax(x textx.Sparse, out []float64) {
	maxZ := math.Inf(-1)
	for c := range m.W {
		z := m.B[c]
		for kk, f := range x.Idx {
			z += m.W[c][f] * float64(x.Val[kk])
		}
		out[c] = z
		if z > maxZ {
			maxZ = z
		}
	}
	var sum float64
	for c := range out {
		out[c] = math.Exp(out[c] - maxZ)
		sum += out[c]
	}
	for c := range out {
		out[c] /= sum
	}
}

// PredictProba returns the class distribution for one vector.
func (m *LogReg) PredictProba(x textx.Sparse) []float64 {
	out := make([]float64, len(m.Classes))
	m.softmax(x, out)
	return out
}

// Predict returns the argmax class name and its probability.
func (m *LogReg) Predict(x textx.Sparse) (string, float64) {
	p := m.PredictProba(x)
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	return m.Classes[best], p[best]
}

// StratifiedFolds assigns each item to one of n folds, keeping class balance.
func StratifiedFolds(y []int, n int, seed int64) []int {
	byClass := map[int][]int{}
	for i, c := range y {
		byClass[c] = append(byClass[c], i)
	}
	keys := make([]int, 0, len(byClass))
	for c := range byClass {
		keys = append(keys, c)
	}
	sort.Ints(keys)

	rng := rand.New(rand.NewSource(seed))
	fold := make([]int, len(y))
	for _, c := range keys {
		is := byClass[c]
		rng.Shuffle(len(is), func(a, b int) { is[a], is[b] = is[b], is[a] })
		for k, i := range is {
			fold[i] = k % n
		}
	}
	return fold
}
