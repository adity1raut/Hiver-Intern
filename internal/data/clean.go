package data

import (
	"html"
	"regexp"
	"strings"
)

var (
	reURL = regexp.MustCompile(`https?://\S+|\bt\.co/\S+`)
	// TWCS pseudonymises customers as numeric handles (@115712); brand handles stay.
	reAnonHandle = regexp.MustCompile(`@\d{4,}`)
	reWS         = regexp.MustCompile(`\s+`)
)

// Clean normalises tweet text. Casing, punctuation and emoji are kept: the reply
// rubric grades tone, and shouting is signal for the escalation decision.
func Clean(s string) string {
	s = html.UnescapeString(s)
	s = reURL.ReplaceAllString(s, "<url>")
	s = reAnonHandle.ReplaceAllString(s, "")
	s = reWS.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
