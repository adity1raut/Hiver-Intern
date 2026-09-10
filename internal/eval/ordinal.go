package eval

import (
	"math"
	"sort"
)

// OrdinalAgreement compares two raters on the 1-5 rubric. Percent agreement alone
// misleads: 4-vs-5 is not the same failure as 1-vs-5.
type OrdinalAgreement struct {
	N              int     `json:"n"`
	ExactAgreement float64 `json:"exact_agreement"`
	Within1        float64 `json:"within_1"`
	MAE            float64 `json:"mae"`
	Bias           float64 `json:"bias"` // mean(a) - mean(b); >0 means a is generous
	QuadraticKappa float64 `json:"quadratic_weighted_kappa"`
	Spearman       float64 `json:"spearman"`
	MeanA          float64 `json:"mean_a"`
	MeanB          float64 `json:"mean_b"`
}

// CompareOrdinal computes agreement between rater a and rater b.
func CompareOrdinal(a, b []float64, minV, maxV int) OrdinalAgreement {
	o := OrdinalAgreement{N: len(a)}
	if len(a) == 0 || len(a) != len(b) {
		return o
	}
	n := float64(len(a))
	var exact, w1, sae, sumA, sumB float64
	for i := range a {
		d := a[i] - b[i]
		if d == 0 {
			exact++
		}
		if math.Abs(d) <= 1 {
			w1++
		}
		sae += math.Abs(d)
		sumA += a[i]
		sumB += b[i]
	}
	o.ExactAgreement, o.Within1, o.MAE = exact/n, w1/n, sae/n
	o.MeanA, o.MeanB = sumA/n, sumB/n
	o.Bias = o.MeanA - o.MeanB
	o.QuadraticKappa = quadraticKappa(a, b, minV, maxV)
	o.Spearman = spearman(a, b)
	return o
}

func quadraticKappa(a, b []float64, minV, maxV int) float64 {
	k := maxV - minV + 1
	if k < 2 {
		return 0
	}
	obs := make([][]float64, k)
	for i := range obs {
		obs[i] = make([]float64, k)
	}
	ha := make([]float64, k)
	hb := make([]float64, k)
	n := float64(len(a))
	clampIdx := func(v float64) int {
		i := int(math.Round(v)) - minV
		if i < 0 {
			return 0
		}
		if i >= k {
			return k - 1
		}
		return i
	}
	for i := range a {
		ia, ib := clampIdx(a[i]), clampIdx(b[i])
		obs[ia][ib]++
		ha[ia]++
		hb[ib]++
	}
	var num, den float64
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			w := float64((i-j)*(i-j)) / float64((k-1)*(k-1))
			exp := ha[i] * hb[j] / n
			num += w * obs[i][j]
			den += w * exp
		}
	}
	if den == 0 {
		return 1
	}
	return 1 - num/den
}

func spearman(a, b []float64) float64 {
	ra, rb := rankAvg(a), rankAvg(b)
	return pearson(ra, rb)
}

func rankAvg(x []float64) []float64 {
	type p struct {
		v float64
		i int
	}
	s := make([]p, len(x))
	for i, v := range x {
		s[i] = p{v, i}
	}
	sort.Slice(s, func(i, j int) bool { return s[i].v < s[j].v })
	r := make([]float64, len(x))
	for i := 0; i < len(s); {
		j := i
		for j+1 < len(s) && s[j+1].v == s[i].v {
			j++
		}
		avg := float64(i+j)/2 + 1 // ties share the average rank
		for k := i; k <= j; k++ {
			r[s[k].i] = avg
		}
		i = j + 1
	}
	return r
}

func pearson(x, y []float64) float64 {
	n := float64(len(x))
	if n == 0 {
		return 0
	}
	var mx, my float64
	for i := range x {
		mx += x[i]
		my += y[i]
	}
	mx, my = mx/n, my/n
	var num, dx, dy float64
	for i := range x {
		a, b := x[i]-mx, y[i]-my
		num += a * b
		dx += a * a
		dy += b * b
	}
	if dx == 0 || dy == 0 {
		return 0
	}
	return num / math.Sqrt(dx*dy)
}
