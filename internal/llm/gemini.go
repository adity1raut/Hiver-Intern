package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// geminiClient talks to the Google Generative Language API.
type geminiClient struct {
	key  string
	http *http.Client
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiReq struct {
	Contents          []geminiContent  `json:"contents"`
	SystemInstruction *geminiContent   `json:"systemInstruction,omitempty"`
	GenerationConfig  map[string]any   `json:"generationConfig,omitempty"`
	SafetySettings    []map[string]any `json:"safetySettings,omitempty"`
}

type geminiResp struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// Default safety filters block a slice of genuine support tweets (profanity,
// descriptions of conflict), which would bias every metric towards the polite half
// of the corpus. Anything still blocked surfaces as an error rather than a drop.
var geminiSafetyOff = []map[string]any{
	{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
}

func (g *geminiClient) Complete(ctx context.Context, r Request) (string, error) {
	body := geminiReq{
		Contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: r.Prompt}}}},
		GenerationConfig: map[string]any{
			"temperature":     r.Temperature,
			"maxOutputTokens": r.MaxTokens,
		},
		SafetySettings: geminiSafetyOff,
	}
	if r.System != "" {
		body.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: r.System}}}
	}
	b, _ := json.Marshal(body)
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", r.Model)

	var lastErr error
	for attempt := 0; attempt < 6; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(b))
		if err != nil {
			return "", err
		}
		req.Header.Set("content-type", "application/json")
		req.Header.Set("x-goog-api-key", g.key)

		resp, err := g.http.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff(attempt))
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("gemini %d: %s", resp.StatusCode, truncate(string(raw), 200))
			time.Sleep(backoff(attempt + 1)) // free tier is aggressively rate limited
			continue
		}
		var out geminiResp
		if err := json.Unmarshal(raw, &out); err != nil {
			return "", fmt.Errorf("gemini: bad response: %s", truncate(string(raw), 300))
		}
		if resp.StatusCode != 200 || out.Error.Code != 0 {
			return "", fmt.Errorf("gemini %d %s: %s", resp.StatusCode, out.Error.Status, truncate(out.Error.Message, 300))
		}
		if len(out.Candidates) == 0 {
			return "", fmt.Errorf("gemini: no candidates (block reason %q)", out.PromptFeedback.BlockReason)
		}
		var sb strings.Builder
		for _, p := range out.Candidates[0].Content.Parts {
			sb.WriteString(p.Text)
		}
		text := sb.String()
		if strings.TrimSpace(text) == "" {
			// A thinking model can spend the whole budget and return an empty part.
			lastErr = fmt.Errorf("gemini: empty text (finish %q)", out.Candidates[0].FinishReason)
			if mt, ok := body.GenerationConfig["maxOutputTokens"].(int); ok && attempt < 2 {
				body.GenerationConfig["maxOutputTokens"] = mt * 3
				b, _ = json.Marshal(body)
				continue
			}
			return "", lastErr
		}
		return text, nil
	}
	return "", fmt.Errorf("gemini: exhausted retries: %w", lastErr)
}
