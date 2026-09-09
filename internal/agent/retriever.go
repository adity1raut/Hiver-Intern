package agent

import (
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

// Precedent is one historical (customer message -> Delta reply) pair offered to
// the drafting step as evidence of how the brand actually handles this.
type Precedent struct {
	EpisodeID  string  `json:"episode_id"`
	Customer   string  `json:"customer"`
	BrandReply string  `json:"brand_reply"`
	Score      float64 `json:"score"`
}

// Retriever is a TF-IDF nearest-neighbour search over the HISTORY split only.
type Retriever struct {
	vec   textx.Vectorizer
	index *textx.Index
	eps   []data.Episode
}

// NewRetriever indexes the historical episodes.
func NewRetriever(history []data.Episode) *Retriever {
	docs := make([]string, len(history))
	for i, e := range history {
		docs[i] = e.CustomerText
	}
	r := &Retriever{eps: history}
	r.vec.Fit(docs, 2, 60000)
	r.index = textx.NewIndex(r.vec.TransformAll(docs))
	return r
}

// Size is the number of indexed precedents.
func (r *Retriever) Size() int { return len(r.eps) }

// Retrieve returns up to k precedents for a message, de-duplicated by reply
// template. Delta's agents reuse boilerplate heavily ("Please DM us..."), so
// without de-duplication the top-k collapses onto one canned line and the
// drafting step loses all the specific handling detail.
func (r *Retriever) Retrieve(msg string, k int) []Precedent {
	q := r.vec.Transform(msg)
	hits := r.index.Search(q, k*8, nil)

	var out []Precedent
	seen := map[string]bool{}
	for _, h := range hits {
		e := r.eps[h.Doc]
		sig := replySignature(e.BrandReply)
		if seen[sig] {
			continue
		}
		seen[sig] = true
		out = append(out, Precedent{
			EpisodeID:  e.EpisodeID,
			Customer:   e.CustomerText,
			BrandReply: e.BrandReply,
			Score:      h.Score,
		})
		if len(out) >= k {
			break
		}
	}
	return out
}

// TopScore is the similarity of the single best precedent, used by the
// escalation policy as a "do we have any precedent at all" signal.
func (r *Retriever) TopScore(msg string) float64 {
	hits := r.index.Search(r.vec.Transform(msg), 1, nil)
	if len(hits) == 0 {
		return 0
	}
	return hits[0].Score
}

// replySignature collapses a reply to its first few content words plus the
// agent-initials suffix stripped, so template twins hash together.
func replySignature(s string) string {
	toks := textx.Tokenize(s)
	if len(toks) > 6 {
		toks = toks[:6]
	}
	return strings.Join(toks, " ")
}
