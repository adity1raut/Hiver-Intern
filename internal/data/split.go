package data

import "sort"

// Split partitions the corpus by time. The agent may only look at History; it is
// evaluated on EvalPool, which is strictly later. A random split would let it
// retrieve a near-duplicate posted minutes apart in the same incident.
type Split struct {
	History  []Episode
	EvalPool []Episode
}

// TemporalSplit puts the earliest historyFrac of episodes into History.
func TemporalSplit(eps []Episode, historyFrac float64) Split {
	s := make([]Episode, len(eps))
	copy(s, eps)
	sort.SliceStable(s, func(i, j int) bool { return s[i].CreatedAt.Before(s[j].CreatedAt) })
	cut := int(float64(len(s)) * historyFrac)
	return Split{History: s[:cut], EvalPool: s[cut:]}
}
