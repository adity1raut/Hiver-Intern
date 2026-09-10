// Command judge grades every system's replies and emits the blind rating tasks a
// human completes via `cmd/review -mode replies`.
//
//	go run ./cmd/judge
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/agent"
	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/eval"
	"github.com/adity1raut/hiver-support-agent/internal/llm"
)

type goldenItem struct {
	EpisodeID    string `json:"episode_id"`
	CustomerText string `json:"customer_text"`
	DeltaReply   string `json:"delta_reply"`
	GoldIntent   string `json:"gold_intent"`
	GoldRoute    string `json:"gold_route"`
}

type replyTask struct {
	EpisodeID    string `json:"episode_id"`
	System       string `json:"system"`
	CustomerText string `json:"customer_text"`
	Reply        string `json:"reply"`
	DeltaReply   string `json:"delta_reply"`
}

func main() {
	var (
		goldenPath = flag.String("golden", "data/golden/golden_set.jsonl", "golden set")
		resultsDir = flag.String("results", "results", "system outputs")
		outPath    = flag.String("out", "results/judge_scores.jsonl", "judge scores")
		tasksPath  = flag.String("tasks", "data/golden/reply_rating_tasks.jsonl", "blind human rating tasks")
		systems    = flag.String("systems", "trivial-escalate,simple,agent,agent-no-retrieval", "systems to judge")
		model      = flag.String("model", "gemini-2.5-pro", "judge model (deliberately not the agent's model)")
		workers    = flag.Int("workers", 6, "parallel workers")
		humanN     = flag.Int("human-n", 60, "reply-rating tasks to emit for the human")
		seed       = flag.Int64("seed", 23, "rng seed")
		alsoDelta  = flag.Bool("judge-delta", true, "also grade Delta's own historical replies as a ceiling")
		limit      = flag.Int("limit", 0, "grade only the first N replies (0 = all); for smoke-testing")
	)
	flag.Parse()

	golden, err := data.ReadJSONL[goldenItem](*goldenPath)
	if err != nil {
		log.Fatal(err)
	}
	byID := map[string]goldenItem{}
	for _, g := range golden {
		byID[g.EpisodeID] = g
	}

	type job struct {
		episodeID, system, customer, reply, actual string
	}
	var jobs []job
	for _, sys := range strings.Split(*systems, ",") {
		sys = strings.TrimSpace(sys)
		if sys == "" {
			continue
		}
		p := filepath.Join(*resultsDir, sys+".jsonl")
		outs, err := data.ReadJSONL[agent.Output](p)
		if err != nil {
			log.Printf("skipping %s: %v", sys, err)
			continue
		}
		for _, o := range outs {
			g, ok := byID[o.EpisodeID]
			if !ok {
				continue
			}
			jobs = append(jobs, job{o.EpisodeID, sys, g.CustomerText, o.Reply, g.DeltaReply})
		}
	}
	// Delta's own reply, same rubric, as a reference row.
	if *alsoDelta {
		for _, g := range golden {
			jobs = append(jobs, job{g.EpisodeID, "delta-human", g.CustomerText, g.DeltaReply, g.DeltaReply})
		}
	}
	if *limit > 0 && *limit < len(jobs) {
		jobs = jobs[:*limit]
	}
	log.Printf("grading %d replies across %d systems", len(jobs), len(strings.Split(*systems, ","))+1)

	client, err := llm.New("cache/llm_cache.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	j := eval.Judge{Client: client, Model: *model}

	scores, _ := llm.Map(jobs, *workers, "judge", func(_ int, jb job) (eval.JudgeScore, error) {
		// Never hand Delta's reply to the judge as its own reference.
		ref := jb.actual
		if jb.system == "delta-human" {
			ref = "(withheld)"
		}
		return j.Grade(context.Background(), jb.episodeID, jb.system, jb.customer, jb.reply, ref), nil
	})

	nErr := 0
	for _, s := range scores {
		if s.Error != "" {
			nErr++
		}
	}
	if nErr > 0 {
		log.Printf("%d/%d judgements errored", nErr, len(scores))
	}
	if err := data.WriteJSONL(*outPath, scores); err != nil {
		log.Fatal(err)
	}
	summarise(scores)

	// Sampled across systems and shuffled so the rater cannot infer the system
	// from the order.
	var tasks []replyTask
	bySys := map[string][]job{}
	for _, jb := range jobs {
		if jb.system == "delta-human" {
			continue
		}
		bySys[jb.system] = append(bySys[jb.system], jb)
	}
	names := make([]string, 0, len(bySys))
	for n := range bySys {
		names = append(names, n)
	}
	sort.Strings(names)
	rng := rand.New(rand.NewSource(*seed))
	per := *humanN / max(1, len(names))
	for _, n := range names {
		js := bySys[n]
		rng.Shuffle(len(js), func(a, b int) { js[a], js[b] = js[b], js[a] })
		for i := 0; i < per && i < len(js); i++ {
			tasks = append(tasks, replyTask{js[i].episodeID, n, js[i].customer, js[i].reply, js[i].actual})
		}
	}
	rng.Shuffle(len(tasks), func(a, b int) { tasks[a], tasks[b] = tasks[b], tasks[a] })
	os.MkdirAll(filepath.Dir(*tasksPath), 0o755)
	if err := data.WriteJSONL(*tasksPath, tasks); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nwrote %s and %d blind rating tasks -> %s\n", *outPath, len(tasks), *tasksPath)
	fmt.Println("NEXT: `go run ./cmd/review -mode replies` to produce the human side of judge agreement.")
	fmt.Printf("cache: %d entries (%d hits, %d misses)\n", client.Size(), client.Hits, client.Misses)
}

func summarise(scores []eval.JudgeScore) {
	type acc struct {
		n, g, r, t, send, viol int
		comp                   float64
	}
	m := map[string]*acc{}
	for _, s := range scores {
		if s.Error != "" {
			continue
		}
		a := m[s.System]
		if a == nil {
			a = &acc{}
			m[s.System] = a
		}
		a.n++
		a.g += s.Grounding
		a.r += s.Resolution
		a.t += s.Tone
		a.comp += s.Composite
		if s.Sendable {
			a.send++
		}
		if s.Violation != "" {
			a.viol++
		}
	}
	names := make([]string, 0, len(m))
	for n := range m {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Printf("\n%-20s %5s %6s %6s %6s %8s %9s %9s\n", "system", "n", "ground", "resol", "tone", "comp", "sendable", "violation")
	for _, n := range names {
		a := m[n]
		f := float64(a.n)
		fmt.Printf("%-20s %5d %6.2f %6.2f %6.2f %8.2f %8.1f%% %8.1f%%\n", n, a.n,
			float64(a.g)/f, float64(a.r)/f, float64(a.t)/f, a.comp/f,
			100*float64(a.send)/f, 100*float64(a.viol)/f)
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
