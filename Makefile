# Delta Twitter support agent - reproducible pipeline.
#
#   make repro   reproduces every headline number from the committed LLM cache,
#                offline, in a couple of minutes. This is the target a reviewer runs.
#   make all     regenerates everything from scratch. Needs an LLM key and ~30 min.

BRAND        ?= Delta
RAW          ?= data/raw/twcs.csv
AGENT_MODEL  ?= gemini-2.5-flash
LABEL_MODEL  ?= gemini-2.5-pro
JUDGE_MODEL  ?= gemini-2.5-pro
WORKERS      ?= 6

.PHONY: help repro all data fetch-data intents golden labels review audit rate-replies run judge report clean test fmt check

help:
	@echo "make repro    - reproduce headline results from the committed cache (offline, ~2 min)"
	@echo "make data     - rebuild Delta episodes from the raw TWCS csv (~1 min, needs data/raw/twcs.csv)"
	@echo "make all      - full regeneration, needs GEMINI_API_KEY or ANTHROPIC_API_KEY"
	@echo "make review   - open the human annotation console"
	@echo "make test     - unit tests"

# ---------------------------------------------------------------- reproduce
# LLM_PROVIDER=offline makes any cache miss a hard error, so this target either
# reproduces the committed numbers exactly or fails loudly.
repro: data
	LLM_PROVIDER=offline go run ./cmd/run-agent -model $(AGENT_MODEL) -workers $(WORKERS)
	LLM_PROVIDER=offline go run ./cmd/judge -model $(JUDGE_MODEL) -workers $(WORKERS)
	go run ./cmd/report
	@echo
	@echo "Headline numbers are in reports/RESULTS.md"

data: data/processed/$(shell echo $(BRAND) | tr A-Z a-z)_episodes.jsonl

data/processed/%_episodes.jsonl: $(RAW)
	go run ./cmd/build-dataset -brand $(BRAND) -raw $(RAW)

$(RAW):
	@echo "data/raw/twcs.csv is missing. Fetch it with:"
	@echo "  make fetch-data"
	@exit 1

fetch-data:
	mkdir -p data/raw
	curl -L --fail -o data/raw/twcs.csv \
	  "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

# ---------------------------------------------------------------- full run
all: data intents golden labels run judge report

intents:
	go run ./cmd/discover-intents -k 30

golden:
	go run ./cmd/sample-golden -n-random 120 -n-targeted 80

labels:
	go run ./cmd/label-assist -model $(LABEL_MODEL) -workers $(WORKERS)

review:
	go run ./cmd/review -mode labels

audit:
	go run ./cmd/review -mode audit

rate-replies:
	go run ./cmd/review -mode replies

run:
	go run ./cmd/run-agent -model $(AGENT_MODEL) -workers $(WORKERS)

judge:
	go run ./cmd/judge -model $(JUDGE_MODEL) -workers $(WORKERS)

report:
	go run ./cmd/report

# ---------------------------------------------------------------- dev
test:
	go test ./...

fmt:
	gofmt -w ./cmd ./internal

check: fmt
	go vet ./...
	go test ./...

clean:
	rm -f results/*.jsonl results/metrics.json reports/RESULTS.md
