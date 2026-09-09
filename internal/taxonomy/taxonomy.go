// Package taxonomy holds the Delta intent set.
//
// The intents were written by hand from the clustering in cmd/discover-intents.
// The organising principle is what the desk does next, not topic similarity: two
// messages share an intent when they route to the same queue and need the same
// information from the customer.
package taxonomy

import "strings"

// Intent is one label in the taxonomy.
type Intent struct {
	Name        string
	Description string
	Includes    string
	Excludes    string
	// DefaultAction is the baseline handling before per-message rules apply.
	DefaultAction string
}

// Intents is the shipped taxonomy. Order is stable and used for report tables.
var Intents = []Intent{
	{
		Name:          "flight_disruption",
		Description:   "A flight is delayed, cancelled, diverted, or the customer has misconnected, and they need to know what happens next.",
		Includes:      "delays, cancellations, missed/tight connections, being stuck at a gate or on the tarmac, requests to be rebooked, 'where is my plane'.",
		Excludes:      "voluntary changes the customer chose to make (booking_change); asking for money because of a disruption (refund_compensation).",
		DefaultAction: "escalate",
	},
	{
		Name:          "baggage",
		Description:   "Anything about the physical handling of a bag.",
		Includes:      "lost, delayed, missing, damaged or pilfered bags, file reference chasing, gate-checked items, carry-on sizing disputes, baggage fees.",
		Excludes:      "compensation demands for a bag already resolved (refund_compensation).",
		DefaultAction: "escalate",
	},
	{
		Name:          "booking_change",
		Description:   "The customer wants to change, cancel or upgrade a reservation they hold.",
		Includes:      "date/route changes, name corrections, seat assignment and seat swaps, upgrade requests and upgrade-list position, standby, cancelling a ticket.",
		Excludes:      "changes forced by a delay or cancellation (flight_disruption).",
		DefaultAction: "escalate",
	},
	{
		Name:          "refund_compensation",
		Description:   "The customer is asking for money, miles or a voucher back.",
		Includes:      "refunds, travel vouchers, compensation for a disruption or bad experience, fee disputes, duplicate or wrong charges, credit not received.",
		Excludes:      "purely informational questions about the refund policy (general_inquiry).",
		DefaultAction: "escalate",
	},
	{
		Name:          "loyalty_program",
		Description:   "SkyMiles, Medallion status, Sky Club and co-brand card questions.",
		Includes:      "missing miles, status match or qualification, Sky Club access and entry denial, award availability, promotions, SkyMiles account access.",
		Excludes:      "a refund of miles already spent (refund_compensation).",
		DefaultAction: "escalate",
	},
	{
		Name:          "digital_issue",
		Description:   "A Delta digital surface is broken for this customer.",
		Includes:      "app crashes or errors, delta.com failures, online check-in failing, mobile boarding pass not loading, in-flight wifi/entertainment not working, login problems.",
		Excludes:      "wanting a human because the app was confusing but nothing is broken (contact_support).",
		DefaultAction: "auto",
	},
	{
		Name:          "service_complaint",
		Description:   "Dissatisfaction with how Delta or its staff treated the customer, where nothing is operationally broken.",
		Includes:      "rude or unhelpful staff, dirty or broken cabin/seat, boarding chaos, catering, general 'worst airline' venting.",
		Excludes:      "complaints whose substance is a delay (flight_disruption) or a money claim (refund_compensation).",
		DefaultAction: "escalate",
	},
	{
		Name:          "contact_support",
		Description:   "The customer's problem is reaching Delta at all - the channel, not the issue.",
		Includes:      "long hold times, dropped calls, 'nobody answers my DM', asking for a phone number or callback, asking to be moved to DM.",
		Excludes:      "messages that also state a substantive problem - label those by the problem.",
		DefaultAction: "auto",
	},
	{
		Name:          "general_inquiry",
		Description:   "A policy, product or logistics question with no incident behind it.",
		Includes:      "baggage allowance, pet/infant/equipment policy, route and schedule questions, aircraft type, fare rules, visa/document questions.",
		Excludes:      "any question tied to a booking that has gone wrong.",
		DefaultAction: "auto",
	},
	{
		Name:          "praise",
		Description:   "Positive feedback with no request attached.",
		Includes:      "thanks, compliments to named crew, upgrade delight, photos from the flight, brand love.",
		Excludes:      "'thanks but...' messages that carry a complaint or request.",
		DefaultAction: "auto",
	},
	{
		Name:          "other",
		Description:   "Not a support request Delta can act on.",
		Includes:      "spam, jokes, marketing, journalists, third-party mentions, unintelligible fragments, messages aimed at another airline.",
		Excludes:      "anything a support agent could action.",
		DefaultAction: "auto",
	},
}

// Names returns the intent names in taxonomy order.
func Names() []string {
	out := make([]string, len(Intents))
	for i, in := range Intents {
		out[i] = in.Name
	}
	return out
}

// Valid reports whether name is in the taxonomy.
func Valid(name string) bool {
	for _, in := range Intents {
		if in.Name == name {
			return true
		}
	}
	return false
}

// Get returns the intent by name.
func Get(name string) (Intent, bool) {
	for _, in := range Intents {
		if in.Name == name {
			return in, true
		}
	}
	return Intent{}, false
}

// Normalise maps a model's free-text label onto the taxonomy, falling back to
// "other". Models answer with near-misses like "flight delay" often enough to matter.
func Normalise(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "-", "_")
	if Valid(s) {
		return s
	}
	alias := map[string]string{
		"delay": "flight_disruption", "delays": "flight_disruption",
		"cancellation": "flight_disruption", "flight_delay": "flight_disruption",
		"disruption": "flight_disruption", "rebooking": "flight_disruption",
		"luggage": "baggage", "bags": "baggage", "lost_baggage": "baggage",
		"booking": "booking_change", "seat": "booking_change", "upgrade": "booking_change",
		"refund": "refund_compensation", "compensation": "refund_compensation",
		"loyalty": "loyalty_program", "skymiles": "loyalty_program", "miles": "loyalty_program",
		"app": "digital_issue", "website": "digital_issue", "technical": "digital_issue",
		"wifi": "digital_issue", "checkin": "digital_issue", "check_in": "digital_issue",
		"complaint": "service_complaint", "feedback": "service_complaint",
		"contact": "contact_support", "support": "contact_support",
		"inquiry": "general_inquiry", "question": "general_inquiry", "general": "general_inquiry",
		"compliment": "praise", "positive": "praise", "thanks": "praise",
		"unknown": "other", "none": "other", "n/a": "other", "mixed": "other",
	}
	if v, ok := alias[s]; ok {
		return v
	}
	return "other"
}

// PromptBlock renders the taxonomy for an LLM prompt.
func PromptBlock() string {
	var b strings.Builder
	for _, in := range Intents {
		b.WriteString("- " + in.Name + ": " + in.Description + "\n")
		b.WriteString("    include: " + in.Includes + "\n")
		b.WriteString("    exclude: " + in.Excludes + "\n")
	}
	return b.String()
}
