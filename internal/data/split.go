package data

import "sort"

// Split is the leak-free temporal partition of the corpus.
//
// Everything the agent may look at (the retrieval index, few-shot examples,
// intent discovery) comes from History. Everything it is evaluated on comes from
// EvalPool, which is strictly LATER in time. A random split would let the agent
// retrieve a near-duplicate tweet posted minutes apart in the same incident
// (e.g. a mass cancellation), which badly inflates reply quality.
type Split struct {
	History  []Episode
	EvalPool []Episode
}

// TemporalSplit puts the earliest `historyFrac` of episodes (by created_at) into
// History and the rest into EvalPool.
func TemporalSplit(eps []Episode, historyFrac float64) Split {
	s := make([]Episode, len(eps))
	copy(s, eps)
	sort.SliceStable(s, func(i, j int) bool { return s[i].CreatedAt.Before(s[j].CreatedAt) })
	cut := int(float64(len(s)) * historyFrac)
	return Split{History: s[:cut], EvalPool: s[cut:]}
}
