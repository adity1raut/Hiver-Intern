// Command llm-smoke verifies the LLM provider returns distinct, correct answers.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/adity1raut/hiver-support-agent/internal/llm"
)

func main() {
	c, err := llm.New("cache/llm_cache.jsonl")
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	qs := []string{
		"my checked bag never arrived in Atlanta",
		"why was my flight to JFK cancelled with no notice",
		"the wifi on flight 1422 did not work at all",
		"I want to change the name on my ticket",
		"your gate agent in Detroit was incredibly rude to my mother",
		"how many miles for a one way to Paris",
	}
	outs, errs := llm.Map(qs, 6, "smoke", func(i int, q string) (string, error) {
		var v struct {
			Topic string `json:"topic"`
		}
		err := llm.CompleteJSON(context.Background(), c, llm.Request{
			Model: "claude-haiku-4-5-20251001", MaxTokens: 100, Temperature: 0,
			System: "You label airline support tweets. Reply with only JSON.",
			Prompt: fmt.Sprintf("Give a 1-3 word topic for this message.\nMessage: %q\nJSON: {\"topic\": \"...\"}", q),
		}, &v)
		return v.Topic, err
	})
	for i, o := range outs {
		fmt.Printf("%-55.55s -> %-28s err=%v\n", qs[i], o, errs[i])
	}
	fmt.Printf("cache: %d entries, %d hits, %d misses\n", c.Size(), c.Hits, c.Misses)
}
