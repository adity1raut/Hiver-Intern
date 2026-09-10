// Command label-assist produces draft labels for the golden set.
//
// Pass A works down from the taxonomy, pass B up from the customer's need, and a
// third pass adjudicates disagreements. All three see the full thread including
// Delta's replies, which the system under test never does.
//
// Output is a draft. cmd/review turns it into the golden set.
// Protocol in docs/ANNOTATION_GUIDE.md.
//
//	go run ./cmd/label-assist
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/llm"
	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
)

type goldenItem struct {
	EpisodeID    string `json:"episode_id"`
	Stratum      string `json:"stratum"`
	StratumNote  string `json:"stratum_note"`
	CreatedAt    string `json:"created_at"`
	CustomerText string `json:"customer_text"`
	DeltaReply   string `json:"delta_reply"`
	NTurns       int    `json:"n_turns"`

	GoldIntent  string `json:"gold_intent,omitempty"`
	GoldRoute   string `json:"gold_route,omitempty"`
	GoldReason  string `json:"gold_reason,omitempty"`
	LabelSource string `json:"label_source,omitempty"`

	PassAIntent string `json:"pass_a_intent,omitempty"`
	PassARoute  string `json:"pass_a_route,omitempty"`
	PassBIntent string `json:"pass_b_intent,omitempty"`
	PassBRoute  string `json:"pass_b_route,omitempty"`
	Agreed      bool   `json:"passes_agreed"`
	Difficulty  string `json:"difficulty,omitempty"`
}

type labelResp struct {
	Intent string `json:"intent"`
	Route  string `json:"route"`
	Reason string `json:"reason"`
}

func main() {
	var (
		in      = flag.String("in", "data/golden/golden_unlabelled.jsonl", "input")
		epsPath = flag.String("episodes", "data/processed/delta_episodes.jsonl", "episodes (for full threads)")
		out     = flag.String("out", "data/golden/golden_draft.jsonl", "output")
		model   = flag.String("model", "claude-opus-5", "annotation model")
		workers = flag.Int("workers", 8, "parallel workers")
		limit   = flag.Int("limit", 0, "label only the first N items (0 = all); for smoke-testing prompts")
	)
	flag.Parse()

	items, err := data.ReadJSONL[goldenItem](*in)
	if err != nil {
		log.Fatal(err)
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}
	eps, err := data.ReadEpisodes(*epsPath)
	if err != nil {
		log.Fatal(err)
	}
	byID := map[string]data.Episode{}
	for _, e := range eps {
		byID[e.EpisodeID] = e
	}

	client, err := llm.New("cache/llm_cache.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	ctx := context.Background()

	label := func(prompt, system string) func(int, goldenItem) (labelResp, error) {
		return func(i int, it goldenItem) (labelResp, error) {
			var v labelResp
			err := llm.CompleteJSON(ctx, client, llm.Request{
				Model: *model, MaxTokens: 500, Temperature: 0,
				System: system,
				Prompt: fmt.Sprintf(prompt, taxonomy.PromptBlock(), taxonomy.RoutePolicy,
					threadBlock(byID[it.EpisodeID]), it.CustomerText),
			}, &v)
			v.Intent = taxonomy.Normalise(v.Intent)
			v.Route = normRoute(v.Route)
			return v, err
		}
	}

	passA, errA := llm.Map(items, *workers, "pass-A", label(promptA,
		"You are a senior support-operations analyst building a gold-standard evaluation set. Be strict and literal about the taxonomy. Reply with only JSON."))
	passB, errB := llm.Map(items, *workers, "pass-B", label(promptB,
		"You are an experienced Delta social-media support lead. Decide what this customer actually needs. Reply with only JSON."))

	var disagreeIdx []int
	for i := range items {
		if errA[i] != nil || errB[i] != nil {
			log.Printf("%s: pass error a=%v b=%v", items[i].EpisodeID, errA[i], errB[i])
		}
		items[i].PassAIntent, items[i].PassARoute = passA[i].Intent, passA[i].Route
		items[i].PassBIntent, items[i].PassBRoute = passB[i].Intent, passB[i].Route
		items[i].Agreed = passA[i].Intent == passB[i].Intent && passA[i].Route == passB[i].Route
		if items[i].Agreed {
			items[i].GoldIntent, items[i].GoldRoute = passA[i].Intent, passA[i].Route
			items[i].GoldReason = passA[i].Reason
			items[i].LabelSource = "draft:both-passes-agree"
			items[i].Difficulty = "easy"
		} else {
			disagreeIdx = append(disagreeIdx, i)
		}
	}
	log.Printf("passes agreed on %d/%d items; adjudicating %d", len(items)-len(disagreeIdx), len(items), len(disagreeIdx))

	adj, errC := llm.Map(disagreeIdx, *workers, "adjudicate", func(_ int, i int) (labelResp, error) {
		it := items[i]
		var v labelResp
		err := llm.CompleteJSON(ctx, client, llm.Request{
			Model: *model, MaxTokens: 600, Temperature: 0,
			System: "You are adjudicating a disagreement between two annotators on a gold-standard support dataset. Pick the better answer or a third one. Reply with only JSON.",
			Prompt: fmt.Sprintf(promptAdjudicate, taxonomy.PromptBlock(), taxonomy.RoutePolicy,
				threadBlock(byID[it.EpisodeID]), it.CustomerText,
				it.PassAIntent, it.PassARoute, it.PassBIntent, it.PassBRoute),
		}, &v)
		v.Intent = taxonomy.Normalise(v.Intent)
		v.Route = normRoute(v.Route)
		return v, err
	})
	for k, i := range disagreeIdx {
		if errC[k] != nil {
			log.Printf("%s: adjudication failed: %v", items[i].EpisodeID, errC[k])
			items[i].GoldIntent, items[i].GoldRoute = items[i].PassAIntent, "escalate" // fail safe
			items[i].LabelSource = "draft:adjudication-failed"
			items[i].Difficulty = "hard"
			continue
		}
		items[i].GoldIntent, items[i].GoldRoute = adj[k].Intent, adj[k].Route
		items[i].GoldReason = adj[k].Reason
		items[i].LabelSource = "draft:adjudicated"
		items[i].Difficulty = "hard"
	}

	if err := data.WriteJSONL(*out, items); err != nil {
		log.Fatal(err)
	}
	summarise(items)
	fmt.Printf("\nwrote %s (cache: %d entries, %d hits, %d misses)\n", *out, client.Size(), client.Hits, client.Misses)
	fmt.Println("NEXT: `go run ./cmd/review` to adjudicate these drafts into the final golden set.")
}

func threadBlock(e data.Episode) string {
	if len(e.Thread) == 0 {
		return "(thread unavailable)"
	}
	var b strings.Builder
	for _, t := range e.Thread {
		who := "CUSTOMER"
		if t.Role == "brand" {
			who = "DELTA"
		}
		fmt.Fprintf(&b, "  %-8s %s\n", who+":", t.Text)
	}
	return b.String()
}

func normRoute(s string) string {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(s)), "auto") {
		return "auto"
	}
	return "escalate"
}

func summarise(items []goldenItem) {
	ic, rc, sc := map[string]int{}, map[string]int{}, map[string]int{}
	agree := 0
	for _, it := range items {
		ic[it.GoldIntent]++
		rc[it.GoldRoute]++
		sc[it.Stratum+"/"+it.GoldRoute]++
		if it.Agreed {
			agree++
		}
	}
	fmt.Printf("\nraw two-pass agreement: %.1f%% (%d/%d)\n", 100*float64(agree)/float64(len(items)), agree, len(items))
	fmt.Println("\nintent distribution:")
	for _, n := range taxonomy.Names() {
		if ic[n] > 0 {
			fmt.Printf("  %-22s %3d\n", n, ic[n])
		}
	}
	fmt.Printf("\nroute: auto=%d escalate=%d\n", rc["auto"], rc["escalate"])
	fmt.Printf("by stratum: random auto=%d esc=%d | targeted auto=%d esc=%d\n",
		sc["random/auto"], sc["random/escalate"], sc["targeted/auto"], sc["targeted/escalate"])
}

const promptA = `INTENT TAXONOMY
%s
%s

FULL CONVERSATION (for context - includes what Delta actually replied and what happened next):
%s
The message to label is the FIRST customer message:
%q

Work top-down from the taxonomy: check each intent's include/exclude list against this
message and pick the single best fit. Then apply the auto/escalate rule.

Reply with only JSON: {"intent": "...", "route": "auto|escalate", "reason": "<one sentence>"}`

const promptB = `INTENT TAXONOMY
%s
%s

FULL CONVERSATION (for context - includes what Delta actually replied and what happened next):
%s
The message to label is the FIRST customer message:
%q

Start from the customer: what do they actually want to happen as a result of sending
this? Then name the taxonomy intent that matches that need, and decide whether a
generic public reply could deliver it.

Reply with only JSON: {"intent": "...", "route": "auto|escalate", "reason": "<one sentence>"}`

const promptAdjudicate = `INTENT TAXONOMY
%s
%s

FULL CONVERSATION:
%s
Message being labelled:
%q

Annotator A said: intent=%s route=%s
Annotator B said: intent=%s route=%s

They disagree. Decide the correct label. You may choose a third option if both are wrong.

Reply with only JSON: {"intent": "...", "route": "auto|escalate", "reason": "<one sentence explaining the disagreement and your call>"}`
