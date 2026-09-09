// Command build-dataset turns the raw TWCS csv into per-brand support episodes.
//
//	go run ./cmd/build-dataset -brand Delta
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/adity1raut/hiver-support-agent/internal/data"
)

const twitterLayout = "Mon Jan 02 15:04:05 -0700 2006"

type tweet struct {
	id        int64
	author    string
	inbound   bool
	createdAt time.Time
	text      string
	parent    int64 // 0 = none
}

func main() {
	var (
		raw    = flag.String("raw", "data/raw/twcs.csv", "path to twcs.csv")
		brand  = flag.String("brand", "Delta", "brand handle to extract")
		outDir = flag.String("out", "data/processed", "output directory")
		minLen = flag.Int("min-len", 15, "minimum characters for customer msg and reply")
	)
	flag.Parse()

	tweets, err := readTweets(*raw)
	if err != nil {
		log.Fatalf("read tweets: %v", err)
	}
	log.Printf("loaded %d tweets", len(tweets))

	eps := buildEpisodes(tweets, *brand, *minLen)
	sort.Slice(eps, func(i, j int) bool { return eps[i].CreatedAt.Before(eps[j].CreatedAt) })

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	out := filepath.Join(*outDir, strings.ToLower(*brand)+"_episodes.jsonl")
	if err := data.WriteEpisodes(out, eps); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s: %d episodes -> %s\n", *brand, len(eps), out)
	if len(eps) > 0 {
		fmt.Printf("date range: %s -> %s\n",
			eps[0].CreatedAt.Format(time.RFC3339), eps[len(eps)-1].CreatedAt.Format(time.RFC3339))
	}
}

func readTweets(path string) (map[int64]*tweet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1 // a handful of rows are ragged
	r.LazyQuotes = true
	r.ReuseRecord = true

	head, err := r.Read()
	if err != nil {
		return nil, err
	}
	col := map[string]int{}
	for i, h := range head {
		col[strings.TrimSpace(h)] = i
	}

	out := make(map[int64]*tweet, 3_000_000)
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // a 3M-row messy csv has ragged rows; skip rather than abort
		}
		if len(rec) < 7 {
			continue
		}
		id, err := strconv.ParseInt(rec[col["tweet_id"]], 10, 64)
		if err != nil {
			continue
		}
		var parent int64
		if p := strings.TrimSpace(rec[col["in_response_to_tweet_id"]]); p != "" {
			if v, err := strconv.ParseFloat(p, 64); err == nil {
				parent = int64(v)
			}
		}
		ts, _ := time.Parse(twitterLayout, rec[col["created_at"]])
		out[id] = &tweet{
			id:        id,
			author:    rec[col["author_id"]],
			inbound:   strings.EqualFold(strings.TrimSpace(rec[col["inbound"]]), "true"),
			createdAt: ts,
			text:      rec[col["text"]],
			parent:    parent,
		}
	}
	return out, nil
}

// rootOf walks in_response_to_tweet_id up to the thread root. The dataset
// contains self-referential rows, hence the cycle and depth guards.
func rootOf(tweets map[int64]*tweet, id int64, memo map[int64]int64) int64 {
	if r, ok := memo[id]; ok {
		return r
	}
	seen := map[int64]bool{}
	cur := id
	for depth := 0; depth < 64; depth++ {
		t, ok := tweets[cur]
		if !ok || t.parent == 0 || seen[cur] {
			break
		}
		if _, ok := tweets[t.parent]; !ok {
			break // dangling parent: treat current as the root
		}
		seen[cur] = true
		cur = t.parent
	}
	memo[id] = cur
	return cur
}

func buildEpisodes(tweets map[int64]*tweet, brand string, minLen int) []data.Episode {
	memo := make(map[int64]int64, len(tweets))
	threads := map[int64][]*tweet{}
	brandRoots := map[int64]bool{}

	for id, t := range tweets {
		r := rootOf(tweets, id, memo)
		threads[r] = append(threads[r], t)
		if t.author == brand {
			brandRoots[r] = true
		}
	}

	var eps []data.Episode
	for root := range brandRoots {
		turns := threads[root]
		sort.Slice(turns, func(i, j int) bool {
			if turns[i].createdAt.Equal(turns[j].createdAt) {
				return turns[i].id < turns[j].id
			}
			return turns[i].createdAt.Before(turns[j].createdAt)
		})
		first := turns[0]
		if !first.inbound {
			continue // brand-initiated, not an inbound support request
		}
		var reply *tweet
		for _, t := range turns[1:] {
			if t.author == brand {
				reply = t
				break
			}
		}
		if reply == nil {
			continue
		}
		cust, rep := data.Clean(first.text), data.Clean(reply.text)
		if len([]rune(cust)) < minLen || len([]rune(rep)) < minLen {
			continue
		}
		ep := data.Episode{
			EpisodeID:    fmt.Sprintf("%s-%d", brand, first.id),
			Brand:        brand,
			CustomerID:   first.author,
			CreatedAt:    first.createdAt,
			CustomerText: cust,
			BrandReply:   rep,
			NTurns:       len(turns),
		}
		for _, t := range turns {
			role := "customer"
			if t.author == brand {
				role = "brand"
				ep.NBrandTurns++
			} else if t.inbound {
				ep.NCustomerTurns++
			}
			ep.Thread = append(ep.Thread, data.Turn{Role: role, Text: data.Clean(t.text)})
		}
		eps = append(eps, ep)
	}
	return eps
}
