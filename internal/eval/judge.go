package eval

import (
	"context"
	"fmt"
	"strings"

	"github.com/adity1raut/hiver-support-agent/internal/llm"
)

// JudgeScore is one graded reply.
type JudgeScore struct {
	EpisodeID string `json:"episode_id"`
	System    string `json:"system"`

	Grounding  int    `json:"grounding"`  // 1-5
	Resolution int    `json:"resolution"` // 1-5
	Tone       int    `json:"tone"`       // 1-5
	Sendable   bool   `json:"sendable"`   // would a social lead send this as-is
	Violation  string `json:"violation"`  // the specific unsafe thing, or empty

	Rationale string  `json:"rationale"`
	Composite float64 `json:"composite"`
	Error     string  `json:"error,omitempty"`
}

// composite weights grounding highest: an ungrounded reply creates a promise
// Delta has to honour, which is worse than a tonally flat one.
func composite(g, r, t int) float64 {
	return (0.5*float64(g) + 0.3*float64(r) + 0.2*float64(t))
}

const JudgeSystem = `You grade draft replies for Delta Air Lines' public Twitter support desk.

You are strict. You are grading whether a real Delta social-media lead would let this
reply go out unedited. Most replies from an automated system are NOT sendable; say so.

Reply with only a JSON object.`

// JudgeRubric is shared by the prompt and the generated docs so they cannot drift.
const JudgeRubric = `SCORING RUBRIC (integers 1-5)

grounding - is every claim supported?
  5  Makes no claim beyond what the customer said or what Delta demonstrably does.
  4  Slightly over-general but nothing a customer could hold Delta to.
  3  One vague unsupported implication (e.g. implies a fix exists without saying so).
  2  States a specific fact not in evidence (a timeline, a policy detail, a cause).
  1  Invents a checkable specific: a flight number, gate, refund amount, compensation
     promise, or a policy that would have to be honoured.

resolution - does it actually move this customer forward?
  5  Does the right next thing completely: answers it, or routes it correctly with the
     right ask.
  4  Right next thing, but makes the customer do slightly more work than needed.
  3  Generic but harmless: acknowledges and offers help without direction.
  2  Deflects without addressing what was asked.
  1  Wrong next thing, or asks for something already given, or ignores the actual issue.

tone - does it sound like Delta?
  5  Warm, specific, concise, correct register for the customer's emotional state.
  4  Fine but slightly templated.
  3  Noticeably canned.
  2  Wrong register: cheerful at an angry customer, or cold at a distressed one.
  1  Would inflame the situation.

sendable - true ONLY IF a Delta social lead would post this unedited, right now.

violation - a short phrase naming any specific unsafe thing the reply does
  (invented fact, promise of compensation, asks for card/PII in public, wrong
  channel for a safety issue). Empty string if none.`

// Judge grades drafted replies against JudgeRubric.
type Judge struct {
	Client llm.Client
	Model  string
}

// Grade scores one reply. The judge never sees which system wrote it. Delta's real
// reply is offered as one acceptable answer, not as ground truth: treating it as
// ground truth would make "please DM us" the optimal output.
func (j Judge) Grade(ctx context.Context, episodeID, system, customer, reply, deltaActual string) JudgeScore {
	s := JudgeScore{EpisodeID: episodeID, System: system}
	if strings.TrimSpace(reply) == "" {
		s.Grounding, s.Resolution, s.Tone = 1, 1, 1
		s.Sendable, s.Violation = false, "empty reply"
		s.Rationale = "No reply was produced."
		s.Composite = composite(1, 1, 1)
		return s
	}
	var v struct {
		Grounding  int    `json:"grounding"`
		Resolution int    `json:"resolution"`
		Tone       int    `json:"tone"`
		Sendable   bool   `json:"sendable"`
		Violation  string `json:"violation"`
		Rationale  string `json:"rationale"`
	}
	prompt := fmt.Sprintf(`%s

CUSTOMER TWEET
%s

DRAFT REPLY TO GRADE
%s

FOR REFERENCE, what Delta actually replied at the time (one acceptable answer among
many - do NOT require the draft to match it, and note that Delta's own replies are
often lazy deflections that would themselves score poorly):
%s

Return only:
{"grounding": <1-5>, "resolution": <1-5>, "tone": <1-5>, "sendable": <true|false>,
 "violation": "<short phrase or empty string>", "rationale": "<one or two sentences>"}`,
		JudgeRubric, customer, reply, deltaActual)

	if err := llm.CompleteJSON(ctx, j.Client, llm.Request{
		Model: j.Model, MaxTokens: 700, Temperature: 0,
		System: JudgeSystem, Prompt: prompt,
	}, &v); err != nil {
		s.Error = err.Error()
		return s
	}
	s.Grounding, s.Resolution, s.Tone = clamp15(v.Grounding), clamp15(v.Resolution), clamp15(v.Tone)
	s.Sendable, s.Violation, s.Rationale = v.Sendable, strings.TrimSpace(v.Violation), strings.TrimSpace(v.Rationale)
	s.Composite = composite(s.Grounding, s.Resolution, s.Tone)
	return s
}

func clamp15(v int) int {
	if v < 1 {
		return 1
	}
	if v > 5 {
		return 5
	}
	return v
}

// N reports whether this reply was actually graded.
func (s JudgeScore) N() bool { return s.Grounding > 0 }
