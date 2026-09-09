// Command sample-golden draws the golden evaluation set from the EVAL POOL
// (the later 30% of Delta's history, never seen by retrieval).
//
// Two strata, kept separate on purpose:
//
//	random   - a uniform sample. Metrics on this stratum alone are an unbiased
//	           estimate of live performance on Delta's real inbound mix.
//	targeted - a stress sample that deliberately over-represents guardrail-tripping
//	           messages (safety, legal, discrimination, money, PII) and rare topical
//	           clusters. Metrics here answer "does it break where breaking is expensive",
//	           and are NOT a estimate of live performance.
//
//	go run ./cmd/sample-golden -n-random 120 -n-targeted 80
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"regexp"
	"sort"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/agent"
	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

// GoldenItem is one row of the golden set. Label fields are filled in by the
// annotation pass (cmd/label-assist + cmd/review).
type GoldenItem struct {
	EpisodeID    string `json:"episode_id"`
	Stratum      string `json:"stratum"`
	StratumNote  string `json:"stratum_note"`
	CreatedAt    string `json:"created_at"`
	CustomerText string `json:"customer_text"`
	DeltaReply   string `json:"delta_reply"` // what Delta actually sent (reference, not a label)
	NTurns       int    `json:"n_turns"`

	GoldIntent  string `json:"gold_intent,omitempty"`
	GoldRoute   string `json:"gold_route,omitempty"`
	GoldReason  string `json:"gold_reason,omitempty"`
	LabelSource string `json:"label_source,omitempty"`
}

var reHasLetters = regexp.MustCompile(`[A-Za-z]{3,}`)

func main() {
	var (
		in        = flag.String("in", "data/processed/delta_episodes.jsonl", "episodes")
		out       = flag.String("out", "data/golden/golden_unlabelled.jsonl", "output")
		nRandom   = flag.Int("n-random", 120, "size of the uniform stratum")
		nTargeted = flag.Int("n-targeted", 80, "size of the stress stratum")
		clusters  = flag.Int("clusters", 20, "clusters used for topical coverage")
		seed      = flag.Int64("seed", 7, "rng seed")
		histFrac  = flag.Float64("history-frac", 0.70, "temporal history fraction")
	)
	flag.Parse()

	eps, err := data.ReadEpisodes(*in)
	if err != nil {
		log.Fatal(err)
	}
	pool := data.TemporalSplit(eps, *histFrac).EvalPool

	// Drop rows that are not a usable support message at all.
	var clean []data.Episode
	seenText := map[string]bool{}
	for _, e := range pool {
		t := strings.TrimSpace(e.CustomerText)
		if len([]rune(t)) < 25 || !reHasLetters.MatchString(t) {
			continue
		}
		key := strings.ToLower(strings.Join(textx.Tokenize(t), " "))
		if key == "" || seenText[key] {
			continue // near-duplicate blasts (same complaint copy-pasted)
		}
		seenText[key] = true
		clean = append(clean, e)
	}
	log.Printf("eval pool %d -> %d after cleaning", len(pool), len(clean))

	rng := rand.New(rand.NewSource(*seed))
	order := rng.Perm(len(clean))
	taken := map[int]bool{}
	var items []GoldenItem

	// ---- stratum 1: uniform random -------------------------------------
	for _, i := range order {
		if len(items) >= *nRandom {
			break
		}
		taken[i] = true
		items = append(items, newItem(clean[i], "random", "uniform draw from the eval pool"))
	}

	// ---- stratum 2: targeted stress ------------------------------------
	// (a) one bucket per guardrail rule, (b) topical clusters for the rest.
	byRule := map[string][]int{}
	for i, e := range clean {
		if taken[i] {
			continue
		}
		for _, r := range agent.HardRules {
			if r.Matches(e.CustomerText) {
				byRule[r.Name] = append(byRule[r.Name], i)
			}
		}
	}
	ruleNames := make([]string, 0, len(byRule))
	for n := range byRule {
		ruleNames = append(ruleNames, n)
	}
	sort.Strings(ruleNames)

	perRule := (*nTargeted / 2) / max(1, len(ruleNames))
	if perRule < 1 {
		perRule = 1
	}
	for _, n := range ruleNames {
		cand := byRule[n]
		rng.Shuffle(len(cand), func(a, b int) { cand[a], cand[b] = cand[b], cand[a] })
		c := 0
		for _, i := range cand {
			if c >= perRule || len(items) >= *nRandom+*nTargeted {
				break
			}
			if taken[i] {
				continue
			}
			taken[i] = true
			items = append(items, newItem(clean[i], "targeted", "guardrail rule: "+n))
			c++
		}
	}

	// topical coverage over the remainder
	if len(items) < *nRandom+*nTargeted {
		docs := make([]string, len(clean))
		for i, e := range clean {
			docs[i] = e.CustomerText
		}
		var vec textx.Vectorizer
		vec.Fit(docs, 3, 20000)
		km := &textx.KMeans{K: *clusters, MaxIter: 30, Seed: *seed}
		km.Fit(vec.TransformAll(docs), vec.Dim())

		members := make([][]int, *clusters)
		for i, c := range km.Labels {
			if !taken[i] {
				members[c] = append(members[c], i)
			}
		}
		for round := 0; len(items) < *nRandom+*nTargeted && round < 50; round++ {
			progressed := false
			for c := 0; c < *clusters && len(items) < *nRandom+*nTargeted; c++ {
				if round >= len(members[c]) {
					continue
				}
				i := members[c][round]
				if taken[i] {
					continue
				}
				taken[i] = true
				items = append(items, newItem(clean[i], "targeted", fmt.Sprintf("topical cluster %d", c)))
				progressed = true
			}
			if !progressed {
				break
			}
		}
	}

	sort.Slice(items, func(a, b int) bool { return items[a].EpisodeID < items[b].EpisodeID })
	if err := data.WriteJSONL(*out, items); err != nil {
		log.Fatal(err)
	}

	counts := map[string]int{}
	for _, it := range items {
		counts[it.Stratum]++
	}
	fmt.Printf("wrote %d golden candidates -> %s\n", len(items), *out)
	fmt.Printf("  random   : %d\n  targeted : %d\n", counts["random"], counts["targeted"])
}

func newItem(e data.Episode, stratum, note string) GoldenItem {
	return GoldenItem{
		EpisodeID: e.EpisodeID, Stratum: stratum, StratumNote: note,
		CreatedAt:    e.CreatedAt.Format("2006-01-02T15:04:05Z"),
		CustomerText: e.CustomerText, DeltaReply: e.BrandReply, NTurns: e.NTurns,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
