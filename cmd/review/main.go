// Command review is the human-in-the-loop annotation console. The distinction
// between its modes is the whole point:
//
//	labels  - review every draft label with the draft shown. Fast, but anchored:
//	          the override rate it yields is a lower bound on true disagreement.
//	audit   - label a random subset cold. The only unbiased measure of draft quality.
//	replies - grade replies on the judge's rubric, blind to the system and to the
//	          judge's scores. The human side of the judge-agreement evidence.
//
// Progress is saved after every item, so it is safe to stop and resume.
//
//	go run ./cmd/review -mode labels
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/data"
	"github.com/adity1raut/hiver-support-agent/internal/eval"
	"github.com/adity1raut/hiver-support-agent/internal/taxonomy"
)

const (
	bold  = "\033[1m"
	dim   = "\033[2m"
	red   = "\033[31m"
	grn   = "\033[32m"
	yel   = "\033[33m"
	cyn   = "\033[36m"
	reset = "\033[0m"
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

	// filled in by cmd/review
	DraftIntent  string `json:"draft_intent,omitempty"`
	DraftRoute   string `json:"draft_route,omitempty"`
	HumanIntent  string `json:"human_intent,omitempty"`
	HumanRoute   string `json:"human_route,omitempty"`
	HumanChanged bool   `json:"human_changed,omitempty"`
	Reviewed     bool   `json:"reviewed,omitempty"`
}

type replyRating struct {
	EpisodeID  string `json:"episode_id"`
	System     string `json:"system"`
	Grounding  int    `json:"grounding"`
	Resolution int    `json:"resolution"`
	Tone       int    `json:"tone"`
	Sendable   bool   `json:"sendable"`
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
		mode      = flag.String("mode", "labels", "labels | audit | replies")
		draft     = flag.String("draft", "data/golden/golden_draft.jsonl", "draft golden labels")
		final     = flag.String("out", "data/golden/golden_set.jsonl", "final golden set")
		auditOut  = flag.String("audit-out", "data/golden/human_audit.jsonl", "blind audit output")
		tasksIn   = flag.String("tasks", "data/golden/reply_rating_tasks.jsonl", "blind reply rating tasks")
		ratingOut = flag.String("ratings", "data/golden/human_reply_ratings.jsonl", "human reply ratings")
		auditN    = flag.Int("audit-n", 40, "items in the blind audit")
		seed      = flag.Int64("seed", 11, "rng seed for audit sampling")
	)
	flag.Parse()

	switch *mode {
	case "labels":
		reviewLabels(*draft, *final)
	case "audit":
		auditLabels(*draft, *auditOut, *auditN, *seed)
	case "replies":
		rateReplies(*tasksIn, *ratingOut)
	default:
		fmt.Fprintln(os.Stderr, "unknown -mode; use labels | audit | replies")
		os.Exit(2)
	}
}

// ------------------------------------------------------------------ labels

func reviewLabels(draftPath, outPath string) {
	items, err := data.ReadJSONL[goldenItem](draftPath)
	must(err)
	// resume from a partially completed run
	if done, err := data.ReadJSONL[goldenItem](outPath); err == nil {
		byID := map[string]goldenItem{}
		for _, d := range done {
			byID[d.EpisodeID] = d
		}
		for i := range items {
			if d, ok := byID[items[i].EpisodeID]; ok && d.Reviewed {
				items[i] = d
			}
		}
	}
	for i := range items {
		if items[i].DraftIntent == "" {
			items[i].DraftIntent, items[i].DraftRoute = items[i].GoldIntent, items[i].GoldRoute
		}
	}

	in := bufio.NewReader(os.Stdin)
	printLabelHelp()

	i := 0
	for i < len(items) {
		it := &items[i]
		if it.Reviewed {
			i++
			continue
		}
		remaining := 0
		for _, x := range items {
			if !x.Reviewed {
				remaining++
			}
		}
		fmt.Printf("\n%s%s[%d/%d  %d left]%s  %s  %s(%s / %s)%s\n",
			bold, cyn, i+1, len(items), remaining, reset, it.EpisodeID, dim, it.Stratum, it.StratumNote, reset)
		fmt.Printf("%sCUSTOMER%s  %s\n", bold, reset, wrap(it.CustomerText, 92, 10))
		fmt.Printf("%sDELTA    %s %s%s%s\n", dim, reset, dim, wrap(it.DeltaReply, 92, 10), reset)
		flag := ""
		if !it.Agreed {
			flag = yel + "  [annotator passes disagreed: A=" + it.PassAIntent + "/" + it.PassARoute +
				"  B=" + it.PassBIntent + "/" + it.PassBRoute + "]" + reset
		}
		fmt.Printf("%sDRAFT%s     %s%s / %s%s%s\n", bold, reset, grn, it.DraftIntent, it.DraftRoute, reset, flag)
		if it.GoldReason != "" {
			fmt.Printf("%s          %s%s\n", dim, wrap(it.GoldReason, 92, 10), reset)
		}
		fmt.Printf("%s> %s", bold, reset)

		line, err := in.ReadString('\n')
		if err != nil {
			break
		}
		cmd := strings.TrimSpace(line)

		switch {
		case cmd == "q":
			save(outPath, items)
			fmt.Printf("\nsaved %s\n", outPath)
			return
		case cmd == "?" || cmd == "h":
			printLabelHelp()
			continue
		case cmd == "b":
			if i > 0 {
				i--
				items[i].Reviewed = false
			}
			continue
		case cmd == "":
			it.HumanIntent, it.HumanRoute = it.DraftIntent, it.DraftRoute
		default:
			ni, nr, ok := parseLabelCmd(cmd, it.DraftIntent, it.DraftRoute)
			if !ok {
				fmt.Printf("%s  ? unrecognised: %q (press ? for help)%s\n", red, cmd, reset)
				continue
			}
			it.HumanIntent, it.HumanRoute = ni, nr
		}
		it.HumanChanged = it.HumanIntent != it.DraftIntent || it.HumanRoute != it.DraftRoute
		it.GoldIntent, it.GoldRoute = it.HumanIntent, it.HumanRoute
		if it.HumanChanged {
			it.LabelSource = "human:overridden"
			fmt.Printf("%s  -> %s / %s (changed)%s\n", yel, it.GoldIntent, it.GoldRoute, reset)
		} else {
			it.LabelSource = "human:confirmed"
		}
		it.Reviewed = true
		save(outPath, items)
		i++
	}
	save(outPath, items)
	summariseLabels(items, outPath)
}

func parseLabelCmd(cmd, curIntent, curRoute string) (string, string, bool) {
	intent, route := curIntent, curRoute
	ok := false
	for _, tok := range strings.Fields(cmd) {
		switch tok {
		case "a":
			route, ok = "auto", true
		case "e":
			route, ok = "escalate", true
		default:
			if n, err := strconv.Atoi(tok); err == nil && n >= 1 && n <= len(taxonomy.Intents) {
				intent, ok = taxonomy.Intents[n-1].Name, true
			} else if taxonomy.Valid(tok) {
				intent, ok = tok, true
			}
		}
	}
	return intent, route, ok
}

func printLabelHelp() {
	fmt.Printf("\n%s%sGOLDEN LABEL REVIEW%s\n", bold, cyn, reset)
	fmt.Println("  ENTER    accept the draft label as-is")
	fmt.Println("  <n>      change intent to n:")
	for i, in := range taxonomy.Intents {
		fmt.Printf("             %s%2d%s %-22s %s\n", bold, i+1, reset, in.Name, dim+truncate(in.Description, 60)+reset)
	}
	fmt.Println("  a | e    set route to auto | escalate")
	fmt.Println("  '3 e'    combine: intent 3 AND escalate")
	fmt.Println("  b        back one item      q  save and quit      ?  this help")
	fmt.Printf("%sJudge by what the message NEEDS, not by how Delta happened to answer it.%s\n", dim, reset)
}

func summariseLabels(items []goldenItem, path string) {
	n, changed := 0, 0
	var draftI, humanI, draftR, humanR []string
	for _, it := range items {
		if !it.Reviewed {
			continue
		}
		n++
		if it.HumanChanged {
			changed++
		}
		draftI = append(draftI, it.DraftIntent)
		humanI = append(humanI, it.HumanIntent)
		draftR = append(draftR, it.DraftRoute)
		humanR = append(humanR, it.HumanRoute)
	}
	fmt.Printf("\n%s%sreviewed %d items, %d overridden (%.1f%%)%s\n", bold, grn, n, changed, 100*float64(changed)/float64(max(1, n)), reset)
	fmt.Printf("  intent  draft-vs-human: agreement %.3f  kappa %.3f\n", eval.Agreement(draftI, humanI), eval.CohenKappa(draftI, humanI))
	fmt.Printf("  route   draft-vs-human: agreement %.3f  kappa %.3f\n", eval.Agreement(draftR, humanR), eval.CohenKappa(draftR, humanR))
	fmt.Printf("  NOTE: anchored review - the draft was visible, so this understates true disagreement.\n")
	fmt.Printf("  saved %s\n", path)
}

// ------------------------------------------------------------------ audit

func auditLabels(draftPath, outPath string, n int, seed int64) {
	items, err := data.ReadJSONL[goldenItem](draftPath)
	must(err)
	rng := rand.New(rand.NewSource(seed))
	order := rng.Perm(len(items))
	if n > len(items) {
		n = len(items)
	}
	pick := make([]goldenItem, 0, n)
	for _, i := range order[:n] {
		pick = append(pick, items[i])
	}
	sort.Slice(pick, func(a, b int) bool { return pick[a].EpisodeID < pick[b].EpisodeID })

	done := map[string]bool{}
	if prev, err := data.ReadJSONL[goldenItem](outPath); err == nil {
		for _, p := range prev {
			done[p.EpisodeID] = true
		}
	}

	in := bufio.NewReader(os.Stdin)
	fmt.Printf("\n%s%sBLIND AUDIT%s - label these cold. The draft label is hidden on purpose:\n", bold, cyn, reset)
	fmt.Println("this is the only unbiased measure of how good the draft labels are.")
	for i, in2 := range taxonomy.Intents {
		fmt.Printf("   %s%2d%s %s\n", bold, i+1, reset, in2.Name)
	}
	fmt.Println("Enter '<intent number> <a|e>', e.g. '3 e'.  q = save and quit")

	var out []goldenItem
	if prev, err := data.ReadJSONL[goldenItem](outPath); err == nil {
		out = prev
	}
	for k, it := range pick {
		if done[it.EpisodeID] {
			continue
		}
		fmt.Printf("\n%s%s[%d/%d]%s %s %s(%s)%s\n", bold, cyn, k+1, len(pick), reset, it.EpisodeID, dim, it.Stratum, reset)
		fmt.Printf("%sCUSTOMER%s  %s\n", bold, reset, wrap(it.CustomerText, 92, 10))
		fmt.Printf("%sDELTA    %s %s%s%s\n", dim, reset, dim, wrap(it.DeltaReply, 92, 10), reset)
		fmt.Printf("%s> %s", bold, reset)
		line, err := in.ReadString('\n')
		if err != nil {
			break
		}
		cmd := strings.TrimSpace(line)
		if cmd == "q" {
			break
		}
		ni, nr, ok := parseLabelCmd(cmd, "", "")
		if !ok || ni == "" {
			fmt.Printf("%s  need an intent number and a|e%s\n", red, reset)
			continue
		}
		rec := it
		rec.HumanIntent, rec.HumanRoute = ni, nr
		rec.Reviewed = true
		out = append(out, rec)
		save(outPath, out)
	}
	// agreement of the blind human labels against the drafts
	var d1, h1, d2, h2 []string
	for _, o := range out {
		d1 = append(d1, o.DraftIntentOrGold())
		h1 = append(h1, o.HumanIntent)
		d2 = append(d2, o.GoldRouteDraft())
		h2 = append(h2, o.HumanRoute)
	}
	fmt.Printf("\n%s%sblind audit: %d items%s\n", bold, grn, len(out), reset)
	fmt.Printf("  intent draft-vs-human: agreement %.3f  kappa %.3f\n", eval.Agreement(d1, h1), eval.CohenKappa(d1, h1))
	fmt.Printf("  route  draft-vs-human: agreement %.3f  kappa %.3f\n", eval.Agreement(d2, h2), eval.CohenKappa(d2, h2))
	fmt.Printf("  saved %s\n", outPath)
}

// DraftIntentOrGold returns the model's draft intent for an audited item.
func (g goldenItem) DraftIntentOrGold() string {
	if g.DraftIntent != "" {
		return g.DraftIntent
	}
	return g.GoldIntent
}

// GoldRouteDraft returns the model's draft route for an audited item.
func (g goldenItem) GoldRouteDraft() string {
	if g.DraftRoute != "" {
		return g.DraftRoute
	}
	return g.GoldRoute
}

// ------------------------------------------------------------------ replies

func rateReplies(tasksPath, outPath string) {
	tasks, err := data.ReadJSONL[replyTask](tasksPath)
	must(err)
	done := map[string]bool{}
	var out []replyRating
	if prev, err := data.ReadJSONL[replyRating](outPath); err == nil {
		out = prev
		for _, p := range prev {
			done[p.EpisodeID+"|"+p.System] = true
		}
	}

	in := bufio.NewReader(os.Stdin)
	fmt.Printf("\n%s%sBLIND REPLY RATING%s\n", bold, cyn, reset)
	fmt.Println("You are scoring drafted replies on the same rubric the LLM judge uses.")
	fmt.Println("Which system wrote each reply, and what the judge scored it, are both hidden:")
	fmt.Println("that is what makes the judge-agreement number meaningful.")
	fmt.Printf("\n%sgrounding%s  5 invents nothing .. 1 invents a checkable specific (flight no, refund amount, policy)\n", bold, reset)
	fmt.Printf("%sresolution%s 5 does the right next thing .. 1 wrong next thing / ignores the ask\n", bold, reset)
	fmt.Printf("%stone%s       5 warm, specific, right register .. 1 would inflame\n", bold, reset)
	fmt.Printf("%ssendable%s   would a Delta social lead post this unedited?\n", bold, reset)
	fmt.Printf("\nEnter four values: %sg r t y|n%s   e.g. %s4 3 4 n%s   (or compact %s434n%s).  s = skip, q = quit\n",
		bold, reset, grn, reset, grn, reset)

	for k, t := range tasks {
		if done[t.EpisodeID+"|"+t.System] {
			continue
		}
		fmt.Printf("\n%s%s[%d/%d]%s %s\n", bold, cyn, k+1, len(tasks), reset, t.EpisodeID)
		fmt.Printf("%sCUSTOMER%s %s\n", bold, reset, wrap(t.CustomerText, 92, 9))
		fmt.Printf("%sREPLY   %s %s%s%s\n", bold, reset, grn, wrap(t.Reply, 92, 9), reset)
		fmt.Printf("%sdelta actually said: %s%s\n", dim, truncate(t.DeltaReply, 150), reset)
		fmt.Printf("%s> %s", bold, reset)
		line, err := in.ReadString('\n')
		if err != nil {
			break
		}
		cmd := strings.TrimSpace(strings.ToLower(line))
		if cmd == "q" {
			break
		}
		if cmd == "s" || cmd == "" {
			continue
		}
		g, r, tn, snd, ok := parseRating(cmd)
		if !ok {
			fmt.Printf("%s  need g r t y|n, e.g. '4 3 4 n'%s\n", red, reset)
			continue
		}
		out = append(out, replyRating{t.EpisodeID, t.System, g, r, tn, snd})
		save(outPath, out)
	}
	fmt.Printf("\n%s%s%d ratings saved -> %s%s\n", bold, grn, len(out), outPath, reset)
}

func parseRating(cmd string) (g, r, t int, sendable, ok bool) {
	cmd = strings.ReplaceAll(cmd, " ", "")
	if len(cmd) < 4 {
		return
	}
	nums := cmd[:3]
	last := cmd[len(cmd)-1]
	for i := 0; i < 3; i++ {
		if nums[i] < '1' || nums[i] > '5' {
			return
		}
	}
	g = int(nums[0] - '0')
	r = int(nums[1] - '0')
	t = int(nums[2] - '0')
	switch last {
	case 'y':
		sendable = true
	case 'n':
		sendable = false
	default:
		return
	}
	return g, r, t, sendable, true
}

// ------------------------------------------------------------------ helpers

func save[T any](path string, items []T) {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return
	}
	enc := json.NewEncoder(f)
	for _, it := range items {
		enc.Encode(it)
	}
	f.Close()
	os.Rename(tmp, path)
}

func wrap(s string, width, indent int) string {
	words := strings.Fields(s)
	var b strings.Builder
	col := 0
	pad := strings.Repeat(" ", indent)
	for _, w := range words {
		if col > 0 && col+len(w)+1 > width {
			b.WriteString("\n" + pad)
			col = 0
		} else if col > 0 {
			b.WriteString(" ")
			col++
		}
		b.WriteString(w)
		col += len(w)
	}
	return b.String()
}

func truncate(s string, n int) string {
	if len([]rune(s)) <= n {
		return s
	}
	return string([]rune(s)[:n]) + "..."
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
