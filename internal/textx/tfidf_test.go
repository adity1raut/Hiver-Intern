package textx

import (
	"math"
	"testing"
)

func TestTokenizeDropsStopwordsAndPunctuation(t *testing.T) {
	got := Tokenize("My flight to ATL was DELAYED 3 hours!!! @Delta")
	want := map[string]bool{"flight": true, "atl": true, "delayed": true, "hours": true}
	for _, g := range got {
		if !want[g] {
			t.Errorf("unexpected token %q in %v", g, got)
		}
	}
	if len(got) != len(want) {
		t.Errorf("got %v, want %d tokens", got, len(want))
	}
}

func TestAnalyzeAddsBigrams(t *testing.T) {
	got := Analyze("lost bag")
	if len(got) != 3 || got[2] != "lost_bag" {
		t.Fatalf("want unigrams + bigram, got %v", got)
	}
}

func TestTransformIsL2Normalised(t *testing.T) {
	var v Vectorizer
	v.Fit([]string{"lost bag atlanta", "delayed flight atlanta", "bag never arrived"}, 1, 0)
	s := v.Transform("lost bag atlanta")
	var n float64
	for _, x := range s.Val {
		n += float64(x) * float64(x)
	}
	if math.Abs(n-1) > 1e-5 {
		t.Fatalf("norm = %v, want 1", n)
	}
}

func TestDotIsCosineSimilarity(t *testing.T) {
	var v Vectorizer
	docs := []string{"my bag is lost", "my bag is missing", "the wifi is broken"}
	v.Fit(docs, 1, 0)
	vs := v.TransformAll(docs)
	same := vs[0].Dot(vs[1])
	diff := vs[0].Dot(vs[2])
	if same <= diff {
		t.Fatalf("similar docs scored %v, dissimilar %v", same, diff)
	}
	if self := vs[0].Dot(vs[0]); math.Abs(self-1) > 1e-5 {
		t.Fatalf("self-similarity %v, want 1", self)
	}
}

func TestIndexFindsNearestNeighbour(t *testing.T) {
	var v Vectorizer
	docs := []string{
		"my checked bag never arrived in atlanta",
		"why was my flight cancelled",
		"the inflight wifi does not work",
	}
	v.Fit(docs, 1, 0)
	ix := NewIndex(v.TransformAll(docs))
	hits := ix.Search(v.Transform("bag never arrived atlanta"), 2, nil)
	if len(hits) == 0 || hits[0].Doc != 0 {
		t.Fatalf("expected doc 0 first, got %+v", hits)
	}
}

func TestEmptyQueryReturnsNothing(t *testing.T) {
	var v Vectorizer
	v.Fit([]string{"lost bag"}, 1, 0)
	ix := NewIndex(v.TransformAll([]string{"lost bag"}))
	if hits := ix.Search(v.Transform("the and of"), 5, nil); len(hits) != 0 {
		t.Fatalf("stopword-only query returned %v", hits)
	}
}
