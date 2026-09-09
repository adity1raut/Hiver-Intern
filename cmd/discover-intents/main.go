// Command discover-intents proposes an intent taxonomy from the data: TF-IDF plus
// spherical k-means over historical messages, then an LLM names each cluster.
//
// The output is a proposal. The shipped taxonomy was written by hand from it.
//
//	go run ./cmd/discover-intents -k 30
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/llm"
	"github.com/adity1raut/hiver-support-agent/internal/textx"
)

type clusterReport struct {
	Cluster   int      `json:"cluster"`
	Size      int      `json:"size"`
	TopTerms  []string `json:"top_terms"`
	Examples  []string `json:"examples"`
	Name      string   `json:"proposed_name"`
	Rationale string   `json:"rationale"`
}

func main() {
	var (
		in     = flag.String("in", "data/processed/delta_episodes.jsonl", "episodes")
		k      = flag.Int("k", 30, "number of clusters")
		sample = flag.Int("sample", 6000, "messages to cluster")
		out    = flag.String("out", "reports/intent_clusters.json", "output")
		model  = flag.String("model", "claude-haiku-4-5-20251001", "model for naming")
		seed   = flag.Int64("seed", 42, "rng seed")
		histFr = flag.Float64("history-frac", 0.70, "temporal history fraction")
	)
	flag.Parse()

	eps, err := data.ReadEpisodes(*in)
	if err != nil {
		log.Fatal(err)
	}
	sp := data.TemporalSplit(eps, *histFr)
	log.Printf("history=%d evalpool=%d", len(sp.History), len(sp.EvalPool))

	// History only: the eval pool must stay unseen.
	rng := rand.New(rand.NewSource(*seed))
	hist := append([]data.Episode(nil), sp.History...)
	rng.Shuffle(len(hist), func(i, j int) { hist[i], hist[j] = hist[j], hist[i] })
	if len(hist) > *sample {
		hist = hist[:*sample]
	}
	docs := make([]string, len(hist))
	for i, e := range hist {
		docs[i] = e.CustomerText
	}

	var vec textx.Vectorizer
	vec.Fit(docs, 3, 30000)
	vecs := vec.TransformAll(docs)
	log.Printf("vocab=%d", vec.Dim())

	km := &textx.KMeans{K: *k, MaxIter: 40, Seed: *seed}
	km.Fit(vecs, vec.Dim())
	terms := km.TopTerms(&vec, 12)

	members := make([][]int, *k)
	for i, c := range km.Labels {
		members[c] = append(members[c], i)
	}

	reps := make([]clusterReport, *k)
	for c := 0; c < *k; c++ {
		r := clusterReport{Cluster: c, Size: len(members[c]), TopTerms: terms[c]}
		for _, i := range members[c] {
			if len(r.Examples) >= 8 {
				break
			}
			r.Examples = append(r.Examples, docs[i])
		}
		reps[c] = r
	}

	client, err := llm.New("cache/llm_cache.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	named, errs := llm.Map(reps, 6, "naming", func(i int, r clusterReport) (clusterReport, error) {
		var v struct {
			Name      string `json:"name"`
			Rationale string `json:"rationale"`
		}
		prompt := fmt.Sprintf(`These are customer tweets sent to Delta Air Lines that clustered together.

Top TF-IDF terms: %s

Example messages:
%s

Give this cluster a short snake_case intent name (2-4 words max) describing what the CUSTOMER WANTS, and one sentence of rationale. If the cluster is incoherent, name it "mixed".
Reply with only JSON: {"name": "...", "rationale": "..."}`,
			strings.Join(r.TopTerms, ", "), "- "+strings.Join(r.Examples, "\n- "))

		err := llm.CompleteJSON(context.Background(), client, llm.Request{
			Model: *model, MaxTokens: 300, Temperature: 0,
			System: "You are a support-operations analyst defining an intent taxonomy. Reply with only JSON.",
			Prompt: prompt,
		}, &v)
		r.Name, r.Rationale = v.Name, v.Rationale
		return r, err
	})
	for i, e := range errs {
		if e != nil {
			log.Printf("cluster %d naming failed: %v", i, e)
		}
	}

	b, _ := json.MarshalIndent(named, "", "  ")
	os.WriteFile(*out, b, 0o644)
	fmt.Printf("\n%-4s %-6s %-34s %s\n", "id", "n", "proposed_name", "top terms")
	for _, r := range named {
		fmt.Printf("%-4d %-6d %-34s %s\n", r.Cluster, r.Size, r.Name, strings.Join(r.TopTerms[:min(6, len(r.TopTerms))], ", "))
	}
	fmt.Printf("\nwrote %s\n", *out)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
