package llm

import "testing"

func TestExtractJSONHandlesFencesAndProse(t *testing.T) {
	cases := []string{
		`{"a":1}`,
		"```json\n{\"a\":1}\n```",
		"Here you go:\n```\n{\"a\":1}\n```\nhope that helps",
		"Sure! {\"a\":1}",
	}
	for _, c := range cases {
		got, err := ExtractJSON(c)
		if err != nil || got != `{"a":1}` {
			t.Errorf("ExtractJSON(%q) = %q, %v", c, got, err)
		}
	}
}

func TestExtractJSONIgnoresBracesInsideStrings(t *testing.T) {
	in := `{"reply":"we use {curly} braces","n":2}`
	got, err := ExtractJSON(in)
	if err != nil || got != in {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestExtractJSONHandlesEscapedQuotes(t *testing.T) {
	in := `{"reply":"he said \"hi\" then left"}`
	got, err := ExtractJSON(in)
	if err != nil || got != in {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestExtractJSONErrorsWhenAbsent(t *testing.T) {
	if _, err := ExtractJSON("no json at all"); err == nil {
		t.Fatal("expected an error")
	}
}

func TestRequestKeyIsStableAndProviderIndependent(t *testing.T) {
	a := Request{Model: "m", Prompt: "p", System: "s", MaxTokens: 10, Temperature: 0}
	b := Request{Model: "m", Prompt: "p", System: "s", MaxTokens: 10, Temperature: 0}
	if a.Key() != b.Key() {
		t.Fatal("identical requests hashed differently")
	}
	c := a
	c.Prompt = "q"
	if a.Key() == c.Key() {
		t.Fatal("different prompts hashed the same")
	}
}
