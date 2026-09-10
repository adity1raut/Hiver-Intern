// Command report computes every headline number into reports/RESULTS.md and
// results/metrics.json.
//
//	go run ./cmd/report
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/agent"
	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/eval"
	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
)

type goldenItem struct {
	EpisodeID    string `json:"episode_id"`
	Stratum      string `json:"stratum"`
	StratumNote  string `json:"stratum_note"`
	CustomerText string `json:"customer_text"`
	DeltaReply   string `json:"delta_reply"`
	GoldIntent   string `json:"gold_intent"`
	GoldRoute    string `json:"gold_route"`
	LabelSource  string `json:"label_source"`
	Difficulty   string `json:"difficulty"`
	Agreed       bool   `json:"passes_agreed"`
	DraftIntent  string `json:"draft_intent"`
	DraftRoute   string `json:"draft_route"`
	HumanChanged bool   `json:"human_changed"`
	Reviewed     bool   `json:"reviewed"`
}

type humanRating struct {
	EpisodeID  string `json:"episode_id"`
	System     string `json:"system"`
	Grounding  int    `json:"grounding"`
	Resolution int    `json:"resolution"`
	Tone       int    `json:"tone"`
	Sendable   bool   `json:"sendable"`
}

type auditItem struct {
	EpisodeID   string `json:"episode_id"`
	DraftIntent string `json:"draft_intent"`
	DraftRoute  string `json:"draft_route"`
	GoldIntent  string `json:"gold_intent"`
	GoldRoute   string `json:"gold_route"`
	HumanIntent string `json:"human_intent"`
	HumanRoute  string `json:"human_route"`
}

// SystemMetrics holds every measurement for one system.
type SystemMetrics struct {
	Name            string              `json:"name"`
	Intent          eval.Classification `json:"intent"`
	IntentRandom    eval.Classification `json:"intent_random_stratum"`
	Routing         eval.Routing        `json:"routing"`
	RoutingRandom   eval.Routing        `json:"routing_random_stratum"`
	RoutingTargeted eval.Routing        `json:"routing_targeted_stratum"`
	Judge           JudgeAgg            `json:"judge"`
	AccuracyCI      [2]float64          `json:"intent_accuracy_ci95"`
	MacroF1CI       [2]float64          `json:"macro_f1_ci95"`
}

// JudgeAgg aggregates judge scores for one system.
type JudgeAgg struct {
	N             int        `json:"n"`
	Grounding     float64    `json:"grounding"`
	Resolution    float64    `json:"resolution"`
	Tone          float64    `json:"tone"`
	Composite     float64    `json:"composite"`
	CompositeCI   [2]float64 `json:"composite_ci95"`
	SendableRate  float64    `json:"sendable_rate"`
	SendableCI    [2]float64 `json:"sendable_ci95"`
	ViolationRate float64    `json:"violation_rate"`
	// AutoSendable restricts sendable to replies actually routed auto, i.e. what a
	// customer would have received.
	AutoSendable  float64 `json:"auto_sendable_rate"`
	AutoSendableN int     `json:"auto_sendable_n"`
}

// Report is the bundle written to results/metrics.json.
type Report struct {
	GoldenSet      GoldenStats     `json:"golden_set"`
	Systems        []SystemMetrics `json:"systems"`
	JudgeAgreement map[string]any  `json:"judge_agreement"`
	LabelQuality   map[string]any  `json:"label_quality"`
	Ablations      map[string]any  `json:"ablations"`
	Calibration    []CalibBin      `json:"calibration"`
}

// GoldenStats describes the evaluation set.
type GoldenStats struct {
	N                 int            `json:"n"`
	ByStratum         map[string]int `json:"by_stratum"`
	IntentCounts      map[string]int `json:"intent_counts"`
	RouteCounts       map[string]int `json:"route_counts"`
	RandomRouteCounts map[string]int `json:"random_stratum_route_counts"`
	Reviewed          int            `json:"human_reviewed"`
	Overridden        int            `json:"human_overridden"`
	TwoPassAgreement  float64        `json:"two_pass_raw_agreement"`
}

// CalibBin is one confidence bucket of the reliability curve.
type CalibBin struct {
	Lo       float64 `json:"lo"`
	Hi       float64 `json:"hi"`
	N        int     `json:"n"`
	MeanConf float64 `json:"mean_confidence"`
	Accuracy float64 `json:"accuracy"`
}

func main() {
	var (
		goldenPath = flag.String("golden", "data/golden/golden_set.jsonl", "golden set")
		resultsDir = flag.String("results", "results", "system outputs")
		judgePath  = flag.String("judge", "results/judge_scores.jsonl", "judge scores")
		humanPath  = flag.String("human", "data/golden/human_reply_ratings.jsonl", "human reply ratings")
		auditPath  = flag.String("audit", "data/golden/human_audit.jsonl", "blind label audit")
		outMD      = flag.String("out-md", "reports/RESULTS.md", "markdown report")
		outJSON    = flag.String("out-json", "results/metrics.json", "metrics json")
		systemsCSV = flag.String("systems", "trivial-escalate,trivial-auto,simple,agent-no-retrieval,agent", "systems in table order")
	)
	flag.Parse()

	golden, err := data.ReadJSONL[goldenItem](*goldenPath)
	if err != nil {
		log.Fatal(err)
	}
	var items []goldenItem
	for _, g := range golden {
		if g.GoldIntent != "" && g.GoldRoute != "" {
			items = append(items, g)
		}
	}
	byID := map[string]goldenItem{}
	for _, g := range items {
		byID[g.EpisodeID] = g
	}

	judgeScores, _ := data.ReadJSONL[eval.JudgeScore](*judgePath)
	judgeBy := map[string]eval.JudgeScore{}
	for _, s := range judgeScores {
		if s.Error == "" {
			judgeBy[s.System+"|"+s.EpisodeID] = s
		}
	}

	rep := Report{
		GoldenSet:      goldenStats(items),
		JudgeAgreement: map[string]any{},
		LabelQuality:   map[string]any{},
		Ablations:      map[string]any{},
	}

	sysNames := splitCSV(*systemsCSV)
	outsBySystem := map[string][]agent.Output{}
	for _, name := range sysNames {
		outs, err := data.ReadJSONL[agent.Output](filepath.Join(*resultsDir, name+".jsonl"))
		if err != nil {
			log.Printf("skipping %s: %v", name, err)
			continue
		}
		outsBySystem[name] = outs
		rep.Systems = append(rep.Systems, systemMetrics(name, outs, byID, judgeBy))
	}
	// Delta's own replies, same rubric.
	if agg, ok := judgeAggFor("delta-human", nil, judgeBy, items); ok {
		rep.Systems = append(rep.Systems, SystemMetrics{Name: "delta-human (reference)", Judge: agg})
	}

	rep.LabelQuality = labelQuality(items, *auditPath)
	rep.JudgeAgreement = judgeAgreement(judgeBy, *humanPath)
	rep.Ablations = ablations(outsBySystem, byID)
	if outs, ok := outsBySystem["agent"]; ok {
		rep.Calibration = calibration(outs, byID)
	}

	os.MkdirAll(filepath.Dir(*outJSON), 0o755)
	b, _ := json.MarshalIndent(rep, "", "  ")
	os.WriteFile(*outJSON, b, 0o644)

	md := renderMarkdown(rep, items, outsBySystem, judgeBy)
	os.MkdirAll(filepath.Dir(*outMD), 0o755)
	os.WriteFile(*outMD, []byte(md), 0o644)

	fmt.Print(md)
	fmt.Printf("\nwrote %s and %s\n", *outMD, *outJSON)
}

func goldenStats(items []goldenItem) GoldenStats {
	s := GoldenStats{
		N: len(items), ByStratum: map[string]int{}, IntentCounts: map[string]int{},
		RouteCounts: map[string]int{}, RandomRouteCounts: map[string]int{},
	}
	agreed := 0
	for _, g := range items {
		s.ByStratum[g.Stratum]++
		s.IntentCounts[g.GoldIntent]++
		s.RouteCounts[g.GoldRoute]++
		if g.Stratum == "random" {
			s.RandomRouteCounts[g.GoldRoute]++
		}
		if g.Reviewed {
			s.Reviewed++
		}
		if g.HumanChanged {
			s.Overridden++
		}
		if g.Agreed {
			agreed++
		}
	}
	if len(items) > 0 {
		s.TwoPassAgreement = float64(agreed) / float64(len(items))
	}
	return s
}

func systemMetrics(name string, outs []agent.Output, byID map[string]goldenItem, judgeBy map[string]eval.JudgeScore) SystemMetrics {
	var gi, pi, gr, pr []string
	var giR, piR, grR, prR []string
	var grT, prT []string
	for _, o := range outs {
		g, ok := byID[o.EpisodeID]
		if !ok {
			continue
		}
		gi = append(gi, g.GoldIntent)
		pi = append(pi, o.Intent)
		gr = append(gr, g.GoldRoute)
		pr = append(pr, string(o.Route))
		if g.Stratum == "random" {
			giR = append(giR, g.GoldIntent)
			piR = append(piR, o.Intent)
			grR = append(grR, g.GoldRoute)
			prR = append(prR, string(o.Route))
		} else {
			grT = append(grT, g.GoldRoute)
			prT = append(prT, string(o.Route))
		}
	}
	labels := taxonomy.Names()
	m := SystemMetrics{
		Name:            name,
		Intent:          eval.Classify(gi, pi, labels),
		IntentRandom:    eval.Classify(giR, piR, labels),
		Routing:         eval.Route(gr, pr),
		RoutingRandom:   eval.Route(grR, prR),
		RoutingTargeted: eval.Route(grT, prT),
	}
	m.AccuracyCI = eval.BootstrapCI(len(gi), 2000, 17, func(idx []int) float64 {
		c := 0
		for _, i := range idx {
			if gi[i] == pi[i] {
				c++
			}
		}
		return float64(c) / float64(len(idx))
	})
	m.MacroF1CI = eval.BootstrapCI(len(gi), 2000, 19, func(idx []int) float64 {
		a := make([]string, len(idx))
		b := make([]string, len(idx))
		for k, i := range idx {
			a[k], b[k] = gi[i], pi[i]
		}
		return eval.Classify(a, b, labels).MacroF1
	})
	if agg, ok := judgeAggFor(name, outs, judgeBy, nil); ok {
		m.Judge = agg
	}
	return m
}

func judgeAggFor(name string, outs []agent.Output, judgeBy map[string]eval.JudgeScore, all []goldenItem) (JudgeAgg, bool) {
	var a JudgeAgg
	var comps []float64
	var sendables []bool
	autoSend, autoN := 0, 0

	consider := func(episodeID string, route agent.Route, hasRoute bool) {
		s, ok := judgeBy[name+"|"+episodeID]
		if !ok {
			return
		}
		a.N++
		a.Grounding += float64(s.Grounding)
		a.Resolution += float64(s.Resolution)
		a.Tone += float64(s.Tone)
		a.Composite += s.Composite
		comps = append(comps, s.Composite)
		sendables = append(sendables, s.Sendable)
		if s.Sendable {
			a.SendableRate++
		}
		if s.Violation != "" {
			a.ViolationRate++
		}
		if hasRoute && route == agent.RouteAuto {
			autoN++
			if s.Sendable {
				autoSend++
			}
		}
	}
	if outs != nil {
		for _, o := range outs {
			consider(o.EpisodeID, o.Route, true)
		}
	} else {
		for _, g := range all {
			consider(g.EpisodeID, "", false)
		}
	}
	if a.N == 0 {
		return a, false
	}
	f := float64(a.N)
	a.Grounding /= f
	a.Resolution /= f
	a.Tone /= f
	a.Composite /= f
	sendCount := int(a.SendableRate)
	a.SendableRate /= f
	a.ViolationRate /= f
	a.SendableCI = eval.WilsonCI(sendCount, a.N)
	a.CompositeCI = eval.BootstrapCI(len(comps), 2000, 29, func(idx []int) float64 {
		s := 0.0
		for _, i := range idx {
			s += comps[i]
		}
		return s / float64(len(idx))
	})
	a.AutoSendableN = autoN
	if autoN > 0 {
		a.AutoSendable = float64(autoSend) / float64(autoN)
	}
	return a, true
}

func labelQuality(items []goldenItem, auditPath string) map[string]any {
	out := map[string]any{}
	var di, hi, dr, hr []string
	reviewed, overridden := 0, 0
	for _, g := range items {
		if !g.Reviewed || g.DraftIntent == "" {
			continue
		}
		reviewed++
		if g.HumanChanged {
			overridden++
		}
		di = append(di, g.DraftIntent)
		hi = append(hi, g.GoldIntent)
		dr = append(dr, g.DraftRoute)
		hr = append(hr, g.GoldRoute)
	}
	out["reviewed"] = reviewed
	out["overridden"] = overridden
	if reviewed > 0 {
		out["override_rate"] = float64(overridden) / float64(reviewed)
		out["anchored_intent_agreement"] = eval.Agreement(di, hi)
		out["anchored_intent_kappa"] = eval.CohenKappa(di, hi)
		out["anchored_route_agreement"] = eval.Agreement(dr, hr)
		out["anchored_route_kappa"] = eval.CohenKappa(dr, hr)
	}
	if audit, err := data.ReadJSONL[auditItem](auditPath); err == nil && len(audit) > 0 {
		var adi, ahi, adr, ahr []string
		for _, a := range audit {
			d := a.DraftIntent
			if d == "" {
				d = a.GoldIntent
			}
			rt := a.DraftRoute
			if rt == "" {
				rt = a.GoldRoute
			}
			adi = append(adi, d)
			ahi = append(ahi, a.HumanIntent)
			adr = append(adr, rt)
			ahr = append(ahr, a.HumanRoute)
		}
		out["blind_audit_n"] = len(audit)
		out["blind_intent_agreement"] = eval.Agreement(adi, ahi)
		out["blind_intent_kappa"] = eval.CohenKappa(adi, ahi)
		out["blind_route_agreement"] = eval.Agreement(adr, ahr)
		out["blind_route_kappa"] = eval.CohenKappa(adr, ahr)
	}
	return out
}

func judgeAgreement(judgeBy map[string]eval.JudgeScore, humanPath string) map[string]any {
	out := map[string]any{}
	human, err := data.ReadJSONL[humanRating](humanPath)
	if err != nil || len(human) == 0 {
		out["status"] = "no human ratings yet - run `go run ./cmd/review -mode replies`"
		return out
	}
	var jg, hg, jr, hr, jt, ht, jc, hc []float64
	var js, hs []string
	for _, h := range human {
		s, ok := judgeBy[h.System+"|"+h.EpisodeID]
		if !ok {
			continue
		}
		jg = append(jg, float64(s.Grounding))
		hg = append(hg, float64(h.Grounding))
		jr = append(jr, float64(s.Resolution))
		hr = append(hr, float64(h.Resolution))
		jt = append(jt, float64(s.Tone))
		ht = append(ht, float64(h.Tone))
		jc = append(jc, s.Composite)
		hc = append(hc, 0.5*float64(h.Grounding)+0.3*float64(h.Resolution)+0.2*float64(h.Tone))
		js = append(js, boolStr(s.Sendable))
		hs = append(hs, boolStr(h.Sendable))
	}
	out["n"] = len(jg)
	if len(jg) == 0 {
		out["status"] = "human ratings did not match any judged reply"
		return out
	}
	out["grounding"] = eval.CompareOrdinal(jg, hg, 1, 5)
	out["resolution"] = eval.CompareOrdinal(jr, hr, 1, 5)
	out["tone"] = eval.CompareOrdinal(jt, ht, 1, 5)
	out["composite_spearman"] = spearmanOf(jc, hc)
	out["sendable_agreement"] = eval.Agreement(js, hs)
	out["sendable_kappa"] = eval.CohenKappa(js, hs)
	// Both are reported: score agreement without sendable agreement is no use.
	return out
}

func spearmanOf(a, b []float64) float64 {
	o := eval.CompareOrdinal(a, b, 1, 5)
	return o.Spearman
}

func ablations(outsBySystem map[string][]agent.Output, byID map[string]goldenItem) map[string]any {
	out := map[string]any{}
	outs, ok := outsBySystem["agent"]
	if !ok {
		return out
	}
	// Route from the model's proposal alone, to isolate the guardrails.
	var gold, withRules, modelOnly []string
	overridden := 0
	for _, o := range outs {
		g, ok := byID[o.EpisodeID]
		if !ok {
			continue
		}
		gold = append(gold, g.GoldRoute)
		withRules = append(withRules, string(o.Route))
		mr := string(o.ModelRoute)
		if mr == "" {
			mr = string(o.Route)
		}
		modelOnly = append(modelOnly, mr)
		if o.Overridden {
			overridden++
		}
	}
	out["routing_with_guardrails"] = eval.Route(gold, withRules)
	out["routing_model_only"] = eval.Route(gold, modelOnly)
	out["guardrail_overrides"] = overridden

	// Rule firing counts and precision against the gold route.
	fired := map[string]int{}
	firedCorrect := map[string]int{}
	for _, o := range outs {
		g, ok := byID[o.EpisodeID]
		if !ok {
			continue
		}
		for _, r := range o.FiredRules {
			fired[r]++
			if g.GoldRoute == "escalate" {
				firedCorrect[r]++
			}
		}
	}
	var rules []ruleRow
	for r, n := range fired {
		rules = append(rules, ruleRow{r, n, float64(firedCorrect[r]) / float64(n)})
	}
	sort.Slice(rules, func(i, j int) bool { return rules[i].Fired > rules[j].Fired })
	out["rule_firing"] = rules
	return out
}

func calibration(outs []agent.Output, byID map[string]goldenItem) []CalibBin {
	edges := []float64{0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001}
	bins := make([]CalibBin, len(edges)-1)
	for i := range bins {
		bins[i].Lo, bins[i].Hi = edges[i], edges[i+1]
	}
	for _, o := range outs {
		g, ok := byID[o.EpisodeID]
		if !ok {
			continue
		}
		for i := range bins {
			if o.Confidence >= edges[i] && o.Confidence < edges[i+1] {
				bins[i].N++
				bins[i].MeanConf += o.Confidence
				if o.Intent == g.GoldIntent {
					bins[i].Accuracy++
				}
				break
			}
		}
	}
	for i := range bins {
		if bins[i].N > 0 {
			bins[i].MeanConf /= float64(bins[i].N)
			bins[i].Accuracy /= float64(bins[i].N)
		}
	}
	return bins
}

func boolStr(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func splitCSV(s string) []string {
	var out []string
	for _, x := range strings.Split(s, ",") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, x)
		}
	}
	return out
}

func pct(f float64) string { return fmt.Sprintf("%.1f%%", 100*f) }

func ci(c [2]float64) string { return fmt.Sprintf("[%.0f-%.0f]", 100*c[0], 100*c[1]) }

func nz(f float64) string {
	if math.IsNaN(f) {
		return "-"
	}
	return fmt.Sprintf("%.3f", f)
}
