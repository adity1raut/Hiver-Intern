// Package textx provides a tokenizer, a TF-IDF vectorizer with inverted-index
// nearest-neighbour search, and spherical k-means. Written from scratch so the
// pipeline builds with no third-party dependencies.
package textx

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// Closed-class English words plus corpus-specific noise that carries no intent signal.
var stopwords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`a about after all also am an and any are as at be because been before being
	but by can cant do does doing dont for from get got had has have he her here hers him his how i if in into is it
	its just me more most my no nor not of off on once only or other our out over own same she should so some such
	than that the their them then there these they this those to too under until up very was we were what when where
	which while who whom why will with would you your yours url delta amp im ive u ur pls plz`) {
		stopwords[w] = true
	}
}

// Tokenize lowercases, splits on non-alphanumeric runes and drops stopwords.
func Tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.Trim(f, "'")
		if len(f) < 2 || stopwords[f] {
			continue
		}
		out = append(out, f)
	}
	return out
}

// Analyze produces unigrams plus bigrams.
func Analyze(s string) []string {
	uni := Tokenize(s)
	out := make([]string, 0, len(uni)*2)
	out = append(out, uni...)
	for i := 0; i+1 < len(uni); i++ {
		out = append(out, uni[i]+"_"+uni[i+1])
	}
	return out
}

// Sparse is an L2-normalised sparse vector, sorted by index.
type Sparse struct {
	Idx []int32
	Val []float32
}

// Dot is the inner product of two sorted sparse vectors.
func (a Sparse) Dot(b Sparse) float64 {
	var s float64
	i, j := 0, 0
	for i < len(a.Idx) && j < len(b.Idx) {
		switch {
		case a.Idx[i] == b.Idx[j]:
			s += float64(a.Val[i]) * float64(b.Val[j])
			i++
			j++
		case a.Idx[i] < b.Idx[j]:
			i++
		default:
			j++
		}
	}
	return s
}

// Vectorizer is TF-IDF over unigrams+bigrams with sublinear TF and L2 normalisation.
type Vectorizer struct {
	Vocab map[string]int32 `json:"vocab"`
	IDF   []float32        `json:"idf"`
	MinDF int              `json:"min_df"`
}

// Fit builds the vocabulary and IDF weights from docs.
func (v *Vectorizer) Fit(docs []string, minDF int, maxFeatures int) {
	df := map[string]int{}
	for _, d := range docs {
		seen := map[string]bool{}
		for _, t := range Analyze(d) {
			if !seen[t] {
				seen[t] = true
				df[t]++
			}
		}
	}
	type kv struct {
		t string
		n int
	}
	kept := make([]kv, 0, len(df))
	for t, n := range df {
		if n >= minDF {
			kept = append(kept, kv{t, n})
		}
	}
	sort.Slice(kept, func(i, j int) bool {
		if kept[i].n != kept[j].n {
			return kept[i].n > kept[j].n
		}
		return kept[i].t < kept[j].t // deterministic tie-break
	})
	if maxFeatures > 0 && len(kept) > maxFeatures {
		kept = kept[:maxFeatures]
	}
	sort.Slice(kept, func(i, j int) bool { return kept[i].t < kept[j].t })

	v.MinDF = minDF
	v.Vocab = make(map[string]int32, len(kept))
	v.IDF = make([]float32, len(kept))
	n := float64(len(docs))
	for i, k := range kept {
		v.Vocab[k.t] = int32(i)
		// smoothed idf, matching sklearn: ln((1+n)/(1+df)) + 1
		v.IDF[i] = float32(math.Log((1+n)/(1+float64(k.n))) + 1)
	}
}

// Transform vectorizes one document.
func (v *Vectorizer) Transform(doc string) Sparse {
	counts := map[int32]float64{}
	for _, t := range Analyze(doc) {
		if i, ok := v.Vocab[t]; ok {
			counts[i]++
		}
	}
	idx := make([]int32, 0, len(counts))
	for i := range counts {
		idx = append(idx, i)
	}
	sort.Slice(idx, func(a, b int) bool { return idx[a] < idx[b] })

	val := make([]float32, len(idx))
	var norm float64
	for k, i := range idx {
		w := (1 + math.Log(counts[i])) * float64(v.IDF[i]) // sublinear tf
		val[k] = float32(w)
		norm += w * w
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for k := range val {
			val[k] *= inv
		}
	}
	return Sparse{Idx: idx, Val: val}
}

// TransformAll vectorizes a corpus.
func (v *Vectorizer) TransformAll(docs []string) []Sparse {
	out := make([]Sparse, len(docs))
	for i, d := range docs {
		out[i] = v.Transform(d)
	}
	return out
}

// Dim is the vocabulary size.
func (v *Vectorizer) Dim() int { return len(v.IDF) }
