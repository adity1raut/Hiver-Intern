package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var reFence = regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)\\s*```")

// ExtractJSON pulls the first JSON object out of a model response, tolerating
// markdown fences and leading prose.
func ExtractJSON(s string) (string, error) {
	s = strings.TrimSpace(s)
	if m := reFence.FindStringSubmatch(s); m != nil {
		s = strings.TrimSpace(m[1])
	}
	start := strings.Index(s, "{")
	if start < 0 {
		return "", fmt.Errorf("no JSON object in response: %s", truncate(s, 200))
	}
	depth, inStr, esc := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case esc:
			esc = false
		case c == '\\' && inStr:
			esc = true
		case c == '"':
			inStr = !inStr
		case inStr:
			// skip
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth == 0 {
				return s[start : i+1], nil
			}
		}
	}
	return "", fmt.Errorf("unbalanced JSON in response: %s", truncate(s, 200))
}

// CompleteJSON completes r and unmarshals the response into v, retrying once
// with a repair instruction if the model returns unparseable JSON.
func CompleteJSON(ctx context.Context, c Client, r Request, v any) error {
	raw, err := c.Complete(ctx, r)
	if err != nil {
		return err
	}
	js, err := ExtractJSON(raw)
	if err == nil {
		if err = json.Unmarshal([]byte(js), v); err == nil {
			return nil
		}
	}
	repair := r
	repair.Prompt = r.Prompt + "\n\nYour previous answer was not valid JSON. Reply with ONLY the JSON object, no prose, no code fence."
	raw2, err2 := c.Complete(ctx, repair)
	if err2 != nil {
		return fmt.Errorf("json parse failed (%v) and repair failed: %w", err, err2)
	}
	js2, err2 := ExtractJSON(raw2)
	if err2 != nil {
		return fmt.Errorf("json parse failed after repair: %w", err2)
	}
	return json.Unmarshal([]byte(js2), v)
}

// Map runs fn over items with `workers` goroutines, preserving input order.
// Errors are returned per item so one bad row cannot sink a whole run.
func Map[In, Out any](items []In, workers int, label string, fn func(int, In) (Out, error)) ([]Out, []error) {
	if workers < 1 {
		workers = 1
	}
	outs := make([]Out, len(items))
	errs := make([]error, len(items))
	var done int64
	start := time.Now()

	var wg sync.WaitGroup
	ch := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				o, err := fn(i, items[i])
				outs[i], errs[i] = o, err
				n := atomic.AddInt64(&done, 1)
				if label != "" && (n%25 == 0 || int(n) == len(items)) {
					log.Printf("%s %d/%d (%.0fs)", label, n, len(items), time.Since(start).Seconds())
				}
			}
		}()
	}
	for i := range items {
		ch <- i
	}
	close(ch)
	wg.Wait()
	return outs, errs
}
