// Command run-agent runs every system over the golden set into results/.
//
//	trivial-escalate     canned reply, always escalate
//	trivial-auto         canned reply, always auto (the unsafe floor)
//	simple               cross-validated logreg intent, nearest-neighbour reply copy, rules
//	agent                retrieval-grounded LLM + the same rules
//	agent-no-retrieval   ablation: precedents removed from the prompt
//
//	go run ./cmd/run-agent
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/adity1raut/hiver-support-agent/internal/agent"
	"github.com/adity1raut/hiver-support-agent/internal/baseline"
	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/llm"
	"github.com/adity1raut/hiver-support-agent/internal/ml"
	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

type goldenItem struct {
	EpisodeID    string `json:"episode_id"`
	Stratum      string `json:"stratum"`
	CustomerText string `json:"customer_text"`
	DeltaReply   string `json:"delta_reply"`
	GoldIntent   string `json:"gold_intent"`
	GoldRoute    string `json:"gold_route"`
}

func main() {
	var (
		goldenPath = flag.String("golden", "data/golden/golden_set.jsonl", "golden set")
		epsPath    = flag.String("episodes", "data/processed/delta_episodes.jsonl", "episodes")
		outDir     = flag.String("out", "results", "output directory")
		model      = flag.String("model", "gemini-2.5-flash", "agent model")
		k          = flag.Int("k", 6, "precedents retrieved per message")
		workers    = flag.Int("workers", 6, "parallel LLM workers")
		folds      = flag.Int("folds", 5, "CV folds for the simple baseline classifier")
		histFrac   = flag.Float64("history-frac", 0.70, "temporal history fraction")
		only       = flag.String("only", "", "comma-separated systems to run (default all)")
		limit      = flag.Int("limit", 0, "run only the first N golden items (0 = all); for smoke-testing")
	)
	flag.Parse()

	golden, err := data.ReadJSONL[goldenItem](*goldenPath)
	if err != nil {
		log.Fatalf("golden set: %v (run cmd/sample-golden, cmd/label-assist, cmd/review first)", err)
	}
	var items []goldenItem
	for _, g := range golden {
		if g.GoldIntent != "" && g.GoldRoute != "" {
			items = append(items, g)
		}
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}
	log.Printf("golden set: %d labelled items", len(items))

	eps, err := data.ReadEpisodes(*epsPath)
	if err != nil {
		log.Fatal(err)
	}
	history := data.TemporalSplit(eps, *histFrac).History
	retr := agent.NewRetriever(history)
	log.Printf("retriever indexed %d historical episodes", retr.Size())

	os.MkdirAll(*outDir, 0o755)
	th := agent.DefaultThresholds
	want := parseOnly(*only)

	majority := baseline.MajorityIntent(intents(items))
	log.Printf("majority intent in the golden set: %s", majority)

	for _, spec := range []struct {
		name  string
		route agent.Route
	}{{"trivial-escalate", agent.RouteEscalate}, {"trivial-auto", agent.RouteAuto}} {
		if !want(spec.name) {
			continue
		}
		t := baseline.Trivial{Intent: majority, Route: spec.route, Reply: baseline.CannedReply, Name: spec.name}
		outs := make([]agent.Output, len(items))
		for i, it := range items {
			outs[i] = t.Run(it.EpisodeID, it.CustomerText)
		}
		writeOut(*outDir, spec.name, outs)
	}

	if want("simple") {
		outs := runSimple(items, retr, th, *folds)
		writeOut(*outDir, "simple", outs)
	}

	needLLM := want("agent") || want("agent-no-retrieval")
	if !needLLM {
		fmt.Println("done (no LLM systems requested)")
		return
	}
	client, err := llm.New("cache/llm_cache.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	for _, spec := range []struct {
		name   string
		noRetr bool
	}{{"agent", false}, {"agent-no-retrieval", true}} {
		if !want(spec.name) {
			continue
		}
		a := &agent.Agent{
			Client: client, Retriever: retr, Model: *model, K: *k,
			Thresholds: th, NoRetrieval: spec.noRetr, Name: spec.name,
		}
		outs, errs := llm.Map(items, *workers, spec.name, func(_ int, it goldenItem) (agent.Output, error) {
			return a.Run(context.Background(), it.EpisodeID, it.CustomerText), nil
		})
		nErr := 0
		for i, o := range outs {
			if o.Error != "" || errs[i] != nil {
				nErr++
			}
		}
		if nErr > 0 {
			log.Printf("%s: %d/%d items errored (they are recorded as forced escalations)", spec.name, nErr, len(items))
		}
		writeOut(*outDir, spec.name, outs)
	}
	fmt.Printf("cache: %d entries (%d hits, %d misses)\n", client.Size(), client.Hits, client.Misses)
}

// runSimple cross-validates the baseline classifier so no item is predicted by a
// model that saw its own label.
func runSimple(items []goldenItem, retr *agent.Retriever, th agent.Thresholds, folds int) []agent.Output {
	docs := make([]string, len(items))
	for i, it := range items {
		docs[i] = it.CustomerText
	}
	var vec textx.Vectorizer
	vec.Fit(docs, 1, 20000)
	X := vec.TransformAll(docs)

	classes := taxonomy.Names()
	ci := map[string]int{}
	for i, c := range classes {
		ci[c] = i
	}
	y := make([]int, len(items))
	for i, it := range items {
		y[i] = ci[it.GoldIntent]
	}

	fold := ml.StratifiedFolds(y, folds, 3)
	outs := make([]agent.Output, len(items))
	for f := 0; f < folds; f++ {
		var trX []textx.Sparse
		var trY []int
		for i := range items {
			if fold[i] != f {
				trX = append(trX, X[i])
				trY = append(trY, y[i])
			}
		}
		clf := &ml.LogReg{LR: 0.5, L2: 1e-4, Epochs: 250, Seed: 5}
		clf.Fit(trX, trY, vec.Dim(), classes)
		s := baseline.Simple{Clf: clf, Vec: &vec, Retriever: retr, Th: th, Name: "simple"}
		for i := range items {
			if fold[i] == f {
				outs[i] = s.Run(items[i].EpisodeID, items[i].CustomerText)
			}
		}
	}
	return outs
}

func intents(items []goldenItem) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.GoldIntent
	}
	return out
}

func writeOut(dir, name string, outs []agent.Output) {
	p := filepath.Join(dir, name+".jsonl")
	if err := data.WriteJSONL(p, outs); err != nil {
		log.Fatal(err)
	}
	auto := 0
	for _, o := range outs {
		if o.Route == agent.RouteAuto {
			auto++
		}
	}
	fmt.Printf("%-20s %4d items  auto=%d (%.0f%%) -> %s\n", name, len(outs), auto,
		100*float64(auto)/float64(len(outs)), p)
}

func parseOnly(s string) func(string) bool {
	if s == "" {
		return func(string) bool { return true }
	}
	set := map[string]bool{}
	for _, x := range splitComma(s) {
		set[x] = true
	}
	return func(n string) bool { return set[n] }
}

func splitComma(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' {
			if cur != "" {
				out = append(out, cur)
			}
			cur = ""
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}
