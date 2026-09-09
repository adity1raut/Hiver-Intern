package textx

import (
	"math"
	"math/rand"
	"sort"
)

// KMeans is spherical k-means with k-means++ seeding. Exploration only: it
// surfaces candidate intent groupings before a human writes the taxonomy.
type KMeans struct {
	K         int
	MaxIter   int
	Seed      int64
	Centroids [][]float32
	Labels    []int
}

func (km *KMeans) Fit(vecs []Sparse, dim int) {
	rng := rand.New(rand.NewSource(km.Seed))
	if km.MaxIter == 0 {
		km.MaxIter = 40
	}
	km.Centroids = kmeansPlusPlus(vecs, dim, km.K, rng)
	km.Labels = make([]int, len(vecs))

	for iter := 0; iter < km.MaxIter; iter++ {
		changed := 0
		for i, v := range vecs {
			best, bestScore := 0, math.Inf(-1)
			for c, cen := range km.Centroids {
				s := dotDense(v, cen)
				if s > bestScore {
					best, bestScore = c, s
				}
			}
			if km.Labels[i] != best {
				changed++
			}
			km.Labels[i] = best
		}
		// recompute centroids as the L2-normalised mean of members
		sums := make([][]float32, km.K)
		counts := make([]int, km.K)
		for c := range sums {
			sums[c] = make([]float32, dim)
		}
		for i, v := range vecs {
			c := km.Labels[i]
			counts[c]++
			for k, idx := range v.Idx {
				sums[c][idx] += v.Val[k]
			}
		}
		for c := range sums {
			if counts[c] == 0 {
				sums[c] = km.Centroids[c]
				continue
			}
			normalize(sums[c])
		}
		km.Centroids = sums
		if changed == 0 {
			break
		}
	}
}

// TopTerms returns the highest-weight vocabulary terms for each cluster.
func (km *KMeans) TopTerms(v *Vectorizer, n int) [][]string {
	inv := make([]string, v.Dim())
	for t, i := range v.Vocab {
		inv[i] = t
	}
	out := make([][]string, km.K)
	for c, cen := range km.Centroids {
		type kv struct {
			t string
			w float32
		}
		all := make([]kv, 0, 256)
		for i, w := range cen {
			if w > 0 {
				all = append(all, kv{inv[i], w})
			}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].w > all[b].w })
		if len(all) > n {
			all = all[:n]
		}
		for _, k := range all {
			out[c] = append(out[c], k.t)
		}
	}
	return out
}

func kmeansPlusPlus(vecs []Sparse, dim, k int, rng *rand.Rand) [][]float32 {
	cens := make([][]float32, 0, k)
	first := rng.Intn(len(vecs))
	cens = append(cens, densify(vecs[first], dim))

	d2 := make([]float64, len(vecs))
	for i := range d2 {
		d2[i] = math.Inf(1)
	}
	for len(cens) < k {
		last := cens[len(cens)-1]
		total := 0.0
		for i, v := range vecs {
			// cosine distance in [0,2]; vectors are unit norm
			dist := 1 - dotDense(v, last)
			if dist < d2[i] {
				d2[i] = dist
			}
			total += d2[i] * d2[i]
		}
		if total == 0 {
			cens = append(cens, densify(vecs[rng.Intn(len(vecs))], dim))
			continue
		}
		target := rng.Float64() * total
		acc, pick := 0.0, len(vecs)-1
		for i := range vecs {
			acc += d2[i] * d2[i]
			if acc >= target {
				pick = i
				break
			}
		}
		cens = append(cens, densify(vecs[pick], dim))
	}
	return cens
}

func densify(v Sparse, dim int) []float32 {
	d := make([]float32, dim)
	for k, i := range v.Idx {
		d[i] = v.Val[k]
	}
	return d
}

func dotDense(v Sparse, c []float32) float64 {
	var s float64
	for k, i := range v.Idx {
		s += float64(v.Val[k]) * float64(c[i])
	}
	return s
}

func normalize(v []float32) {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(n))
	for i := range v {
		v[i] *= inv
	}
}
