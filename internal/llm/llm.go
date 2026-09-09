// Package llm is a provider-agnostic LLM client with an on-disk cache.
//
// Providers: gemini (GEMINI_API_KEY), anthropic (ANTHROPIC_API_KEY), cli (a local
// `claude` binary), offline (cache only).
//
// The cache key hashes the request but not the provider, so a cache built with one
// provider is reused by another. It is committed, which is what lets `make repro`
// run offline with no key.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Request is one completion request.
type Request struct {
	Model       string  `json:"model"`
	System      string  `json:"system,omitempty"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

// Key is the stable cache key for a request.
func (r Request) Key() string {
	b, _ := json.Marshal(r)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// Client completes a request.
type Client interface {
	Complete(ctx context.Context, r Request) (string, error)
}

// ---------------------------------------------------------------- cache

type cacheEntry struct {
	Key      string  `json:"key"`
	Request  Request `json:"request"`
	Response string  `json:"response"`
	Provider string  `json:"provider"`
	At       string  `json:"at"`
}

// Cached wraps a Client with a persistent JSONL cache.
type Cached struct {
	inner    Client
	provider string
	path     string
	mu       sync.Mutex
	mem      map[string]string
	fh       *os.File
	// Offline turns a cache miss into an error instead of a live call.
	Offline bool

	Hits, Misses int
}

// NewCached opens (or creates) the cache at path and wraps inner.
func NewCached(inner Client, provider, path string) (*Cached, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	c := &Cached{inner: inner, provider: provider, path: path, mem: map[string]string{}}
	if f, err := os.Open(path); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 1<<20), 1<<25)
		for sc.Scan() {
			if len(sc.Bytes()) == 0 {
				continue
			}
			var e cacheEntry
			if json.Unmarshal(sc.Bytes(), &e) == nil {
				c.mem[e.Key] = e.Response
			}
		}
		f.Close()
	}
	fh, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	c.fh = fh
	return c, nil
}

// Size reports how many completions are cached.
func (c *Cached) Size() int { c.mu.Lock(); defer c.mu.Unlock(); return len(c.mem) }

// Close flushes the cache file.
func (c *Cached) Close() error { return c.fh.Close() }

// Complete returns a cached response when present, else calls the inner client.
func (c *Cached) Complete(ctx context.Context, r Request) (string, error) {
	k := r.Key()
	c.mu.Lock()
	v, ok := c.mem[k]
	c.mu.Unlock()
	if ok {
		c.mu.Lock()
		c.Hits++
		c.mu.Unlock()
		return v, nil
	}
	if c.Offline {
		return "", fmt.Errorf("llm: cache miss in offline mode (key %s); re-run with LLM_PROVIDER set", k[:12])
	}
	resp, err := c.inner.Complete(ctx, r)
	if err != nil {
		return "", err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Misses++
	c.mem[k] = resp
	b, _ := json.Marshal(cacheEntry{Key: k, Request: r, Response: resp,
		Provider: c.provider, At: time.Now().UTC().Format(time.RFC3339)})
	c.fh.Write(append(b, '\n'))
	c.fh.Sync()
	return resp, nil
}

// ---------------------------------------------------------------- anthropic API

type anthropicClient struct {
	key  string
	http *http.Client
}

func (a *anthropicClient) Complete(ctx context.Context, r Request) (string, error) {
	body := map[string]any{
		"model":       r.Model,
		"max_tokens":  r.MaxTokens,
		"temperature": r.Temperature,
		"messages":    []map[string]string{{"role": "user", "content": r.Prompt}},
	}
	if r.System != "" {
		body["system"] = r.System
	}
	b, _ := json.Marshal(body)

	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST",
			"https://api.anthropic.com/v1/messages", bytes.NewReader(b))
		if err != nil {
			return "", err
		}
		req.Header.Set("x-api-key", a.key)
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("content-type", "application/json")

		resp, err := a.http.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff(attempt))
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("anthropic %d: %s", resp.StatusCode, truncate(string(raw), 200))
			time.Sleep(backoff(attempt))
			continue
		}
		if resp.StatusCode != 200 {
			return "", fmt.Errorf("anthropic %d: %s", resp.StatusCode, truncate(string(raw), 400))
		}
		var out struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(raw, &out); err != nil {
			return "", err
		}
		var sb strings.Builder
		for _, c := range out.Content {
			if c.Type == "text" {
				sb.WriteString(c.Text)
			}
		}
		return sb.String(), nil
	}
	return "", fmt.Errorf("anthropic: exhausted retries: %w", lastErr)
}

// ---------------------------------------------------------------- claude CLI

type cliClient struct{ bin string }

func (c *cliClient) Complete(ctx context.Context, r Request) (string, error) {
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		args := []string{"-p", r.Prompt, "--model", r.Model, "--output-format", "json"}
		if r.System != "" {
			args = append(args, "--system-prompt", r.System)
		}
		cctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
		cmd := exec.CommandContext(cctx, c.bin, args...)
		cmd.Stdin = bytes.NewReader(nil) // avoid the CLI's stdin probe
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err := cmd.Run()
		cancel()
		if err != nil {
			lastErr = fmt.Errorf("%v: %s", err, truncate(stderr.String(), 300))
			time.Sleep(backoff(attempt))
			continue
		}
		var out struct {
			Result  string `json:"result"`
			IsError bool   `json:"is_error"`
		}
		if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
			return strings.TrimSpace(stdout.String()), nil // plain-text fallback
		}
		if out.IsError {
			lastErr = errors.New(truncate(out.Result, 300))
			time.Sleep(backoff(attempt))
			continue
		}
		return out.Result, nil
	}
	return "", fmt.Errorf("claude cli: exhausted retries: %w", lastErr)
}

// ---------------------------------------------------------------- construction

// New builds a cached client. The provider comes from LLM_PROVIDER, else is
// inferred from whichever key or binary is present.
func New(cachePath string) (*Cached, error) {
	provider := os.Getenv("LLM_PROVIDER")
	if provider == "" {
		switch {
		case os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "":
			provider = "gemini"
		case os.Getenv("ANTHROPIC_API_KEY") != "":
			provider = "anthropic"
		case findCLI() != "":
			provider = "cli"
		default:
			provider = "offline"
		}
	}
	var inner Client
	switch provider {
	case "gemini":
		key := os.Getenv("GEMINI_API_KEY")
		if key == "" {
			key = os.Getenv("GOOGLE_API_KEY")
		}
		if key == "" {
			return nil, errors.New("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty")
		}
		inner = &geminiClient{key: key, http: &http.Client{Timeout: 5 * time.Minute}}
	case "anthropic":
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			return nil, errors.New("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
		}
		inner = &anthropicClient{key: key, http: &http.Client{Timeout: 5 * time.Minute}}
	case "cli":
		bin := findCLI()
		if bin == "" {
			return nil, errors.New("LLM_PROVIDER=cli but no `claude` binary found")
		}
		inner = &cliClient{bin: bin}
	case "offline":
		inner = nil
	default:
		return nil, fmt.Errorf("unknown LLM_PROVIDER %q", provider)
	}
	c, err := NewCached(inner, provider, cachePath)
	if err != nil {
		return nil, err
	}
	c.Offline = provider == "offline"
	return c, nil
}

func findCLI() string {
	if p := os.Getenv("CLAUDE_CODE_EXECPATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p, err := exec.LookPath("claude"); err == nil {
		return p
	}
	return ""
}

func backoff(attempt int) time.Duration {
	return time.Duration(1<<attempt) * 1500 * time.Millisecond
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
