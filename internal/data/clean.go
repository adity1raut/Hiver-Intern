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

// Clean normalises tweet text for modelling.
//
// We unescape HTML entities, replace URLs with a <url> placeholder and strip the
// pseudonymised customer handles. We deliberately KEEP casing, punctuation and
// emoji: the reply-quality rubric grades tone, and shouting/emoji are signal for
// the escalation decision.
func Clean(s string) string {
	s = html.UnescapeString(s)
	s = reURL.ReplaceAllString(s, "<url>")
	s = reAnonHandle.ReplaceAllString(s, "")
	s = reWS.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
