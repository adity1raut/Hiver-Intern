package textx

import "sort"

// Index is an inverted index over sparse vectors for exact cosine top-k search.
// With L2-normalised TF-IDF vectors the inner product IS the cosine similarity.
type Index struct {
	postings map[int32][]posting
	n        int
}

type posting struct {
	doc int32
	val float32
}

// Neighbour is one search hit.
type Neighbour struct {
	Doc   int
	Score float64
}

// NewIndex builds an inverted index over vecs.
func NewIndex(vecs []Sparse) *Index {
	ix := &Index{postings: make(map[int32][]posting), n: len(vecs)}
	for d, v := range vecs {
		for k, term := range v.Idx {
			ix.postings[term] = append(ix.postings[term], posting{int32(d), v.Val[k]})
		}
	}
	return ix
}

// Search returns the top-k most cosine-similar documents to q.
// `allow` (optional) filters candidate documents.
func (ix *Index) Search(q Sparse, k int, allow func(int) bool) []Neighbour {
	scores := make(map[int32]float64, 1024)
	for i, term := range q.Idx {
		qv := float64(q.Val[i])
		for _, p := range ix.postings[term] {
			scores[p.doc] += qv * float64(p.val)
		}
	}
	out := make([]Neighbour, 0, len(scores))
	for d, s := range scores {
		if s <= 0 {
			continue
		}
		if allow != nil && !allow(int(d)) {
			continue
		}
		out = append(out, Neighbour{Doc: int(d), Score: s})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].Doc < out[b].Doc // deterministic
	})
	if len(out) > k {
		out = out[:k]
	}
	return out
}
