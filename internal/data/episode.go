// Package data defines the corpus types and the raw-tweet -> episode transform.
package data

import (
	"bufio"
	"encoding/json"
	"os"
	"time"
)

// Turn is one message in a support thread.
type Turn struct {
	Role string `json:"role"` // "customer" | "brand"
	Text string `json:"text"`
}

// Episode is one thread rooted at an inbound customer tweet that the brand replied to.
type Episode struct {
	EpisodeID      string    `json:"episode_id"`
	Brand          string    `json:"brand"`
	CustomerID     string    `json:"customer_id"`
	CreatedAt      time.Time `json:"created_at"`
	CustomerText   string    `json:"customer_text"`
	BrandReply     string    `json:"brand_reply"`
	NTurns         int       `json:"n_turns"`
	NBrandTurns    int       `json:"n_brand_turns"`
	NCustomerTurns int       `json:"n_customer_turns"`
	Thread         []Turn    `json:"thread"`
}

// ReadEpisodes loads a JSONL episode file.
func ReadEpisodes(path string) ([]Episode, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Episode
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var e Episode
		if err := json.Unmarshal(line, &e); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// WriteEpisodes writes episodes as JSONL.
func WriteEpisodes(path string, eps []Episode) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	enc := json.NewEncoder(w)
	for _, e := range eps {
		if err := enc.Encode(e); err != nil {
			return err
		}
	}
	return nil
}

// WriteJSONL writes any slice as JSONL.
func WriteJSONL[T any](path string, items []T) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	enc := json.NewEncoder(w)
	for _, it := range items {
		if err := enc.Encode(it); err != nil {
			return err
		}
	}
	return nil
}

// ReadJSONL reads any JSONL file into a slice.
func ReadJSONL[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var v T
		if err := json.Unmarshal(sc.Bytes(), &v); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, sc.Err()
}
