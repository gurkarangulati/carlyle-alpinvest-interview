# One command per stage. Run from the repo root.  `make help` lists them.
#
# Settings are variables; pass them on the command line or export them:
#   make deploy-aws PROFILE=my-aws-profile        (falls back to $AWS_PROFILE)
#   make data SEC_USER_AGENT="Your Name you@example.com"
#   make deploy-gcp PROJECT=my-gcp-project        (falls back to $GOOGLE_CLOUD_PROJECT)

PROFILE ?= $(AWS_PROFILE)
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
REGION  ?= us-east-1
# C: regularization for the backtest (chosen by CV).
# LLM_TEST: the prompt version chosen on validation, run on test once.
C       ?= 100
LLM_TEST ?= --shots 5
STAMP   := $(shell date +%Y-%m-%d_%H%M)
PY      := uv run python
AWS     := AWS_PROFILE=$(PROFILE) AWS_REGION=$(REGION)

.PHONY: help setup data names xbrl test classic finetune llm-val llm-test llm-gemini \
	    matching matching-llm extraction qa report all package image serve serve-docker \
	    deploy-aws smoke-aws deploy-gcp smoke-gcp teardown-aws teardown-gcp \
	    check-aws check-sec check-gcp

help:  ## list the stages
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

# ---- guards: stop early with a clear message instead of failing halfway ----
check-aws:
	@test -n "$(PROFILE)" || { echo "Set PROFILE=<aws profile> or export AWS_PROFILE"; exit 1; }
check-sec:
	@test -n "$(SEC_USER_AGENT)" || { echo 'Set SEC_USER_AGENT="Your Name you@example.com" (the SEC requires it)'; exit 1; }
check-gcp:
	@test -n "$(PROJECT)" || { echo "Set PROJECT=<gcp project id> or export GOOGLE_CLOUD_PROJECT"; exit 1; }

# ---- setup and data ----
setup:  ## install every package group (core, ml, deploy, dev)
	uv sync --all-groups

data: check-sec  ## download 10-Ks, fetch SIC codes, build data/companies.csv, split by company
	$(PY) fixtures/make_companies.py download
	$(PY) fixtures/make_companies.py texts
	SEC_USER_AGENT="$(SEC_USER_AGENT)" $(PY) fixtures/make_companies.py sic
	$(PY) fixtures/make_companies.py merge
	$(PY) sectors/split.py

names: check-sec  ## SEC current and former names + the matching golden set
	SEC_USER_AGENT="$(SEC_USER_AGENT)" $(PY) -m matching.fetch_sec_names
	$(PY) -m matching.golden

xbrl: check-sec  ## SEC XBRL facts (the answer key for extraction) + the 119-filing sample
	SEC_USER_AGENT="$(SEC_USER_AGENT)" $(PY) extraction/fetch_xbrl.py
	$(PY) extraction/sample.py

test:  ## unit tests
	$(PY) -m pytest -q tests/

# ---- sectors: classic ML, fine-tuning, foundation models ----
classic:  ## split check, TF-IDF baseline, wider C, balanced, 5-fold CV, walk-forward backtest
	$(PY) sectors/split_options.py
	$(PY) sectors/baseline.py
	$(PY) sectors/baseline.py --c 10 30 100 300 1000
	$(PY) sectors/baseline.py --c 10 30 100 300 1000 --balanced
	$(PY) sectors/cv.py --c 10 30 100 300 1000
	$(PY) sectors/backtest.py --c $(C)

finetune:  ## fine-tune DistilRoBERTa: all weights, then LoRA
	$(PY) sectors/finetune.py
	$(PY) sectors/finetune.py --lora

llm-val: check-aws  ## Claude on Bedrock, four prompt versions on VALIDATION
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile val --guide v1
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile val --guide v2
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile val --shots 5
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile val --guide v2 --shots 5

llm-test: check-aws  ## Claude zero-shot + the version chosen on validation, on TEST (once)
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile test
	$(AWS) $(PY) sectors/llm_classify.py --provider bedrock --pile test $(LLM_TEST)

llm-gemini: check-gcp  ## Gemini zero-shot on Vertex AI, on TEST (needs gcloud ADC login)
	GOOGLE_CLOUD_PROJECT=$(PROJECT) $(PY) sectors/llm_classify.py --provider vertex --pile test

report:  ## every classifier run in one table (sectors/out/results.csv)
	$(PY) sectors/report.py

# ---- matching and extraction (console output saved to each module's results/logs) ----
matching:  ## name matching: tune on dev, test once, mentions demo
	mkdir -p matching/results/logs
	$(PY) -m matching.evaluate --split dev 2>&1 | tee matching/results/logs/eval_dev_$(STAMP).txt
	$(PY) -m matching.evaluate --split test 2>&1 | tee matching/results/logs/eval_test_$(STAMP).txt
	$(PY) -m matching.mentions 2>&1 | tee matching/results/logs/mentions_$(STAMP).txt

matching-llm: check-aws  ## name matching with Claude re-ranking close calls (dev)
	mkdir -p matching/results/logs
	$(AWS) $(PY) -m matching.evaluate --split dev --llm-rerank 2>&1 | tee matching/results/logs/eval_dev_llm_$(STAMP).txt

extraction: check-aws  ## number extraction: single-call baseline vs. LangGraph flow, scored on XBRL
	mkdir -p extraction/results/logs
	$(AWS) $(PY) extraction/evaluate.py --variant baseline 2>&1 | tee extraction/results/logs/evaluate_baseline_$(STAMP).txt
	$(AWS) $(PY) extraction/evaluate.py --variant graph 2>&1 | tee extraction/results/logs/evaluate_graph_$(STAMP).txt

qa: check-aws  ## document Q&A: no-tools baseline vs. tool-using agent (run after extraction)
	mkdir -p extraction/results/logs
	$(AWS) $(PY) extraction/qa_eval.py --variant baseline 2>&1 | tee extraction/results/logs/qa_baseline_$(STAMP).txt
	$(AWS) $(PY) extraction/qa_eval.py --variant agent 2>&1 | tee extraction/results/logs/qa_agent_$(STAMP).txt

all: setup data names xbrl test classic finetune llm-val llm-test matching extraction qa report  ## everything except deploys

# ---- serving and deployment ----
package:  ## tar the fine-tuned model for SageMaker (sectors/out/model.tar.gz)
	$(PY) sectors/deploy/package_model.py

image:  ## build the serving container locally (same Dockerfile Cloud Run uses)
	rm -rf sectors/deploy/app/model && cp -R sectors/out/models/finetune_full sectors/deploy/app/model
	docker build -t sector-classifier sectors/deploy/app

serve:  ## run the API locally without Docker: http://localhost:8080/docs
	cd sectors/deploy/app && MODEL_DIR=../../out/models/finetune_full uv run uvicorn main:app --port 8080

serve-docker: image  ## run the container locally: http://localhost:8080/docs
	docker run --rm -p 8080:8080 sector-classifier

deploy-aws: check-aws package  ## API Gateway -> Lambda -> SageMaker serverless, via CDK
	cd sectors/deploy/aws && $(AWS) npx aws-cdk@latest deploy --require-approval broadening

smoke-aws: check-aws  ## call the AWS API: cold, warm, and without a key
	$(AWS) bash sectors/deploy/aws/smoke.sh

deploy-gcp: check-gcp  ## build with Cloud Build, run on Cloud Run (private)
	GOOGLE_CLOUD_PROJECT=$(PROJECT) bash sectors/deploy/gcp/deploy.sh

smoke-gcp: check-gcp  ## call the Cloud Run service: twice with a token, once without
	GOOGLE_CLOUD_PROJECT=$(PROJECT) bash sectors/deploy/gcp/smoke.sh

teardown-aws: check-aws  ## delete the AWS stack (then see sectors/deploy/TEARDOWN.md for logs and assets)
	cd sectors/deploy/aws && $(AWS) npx aws-cdk@latest destroy

teardown-gcp: check-gcp  ## delete the Cloud Run service (then see TEARDOWN.md for the image and source)
	gcloud run services delete sector-classifier --region us-central1 --project $(PROJECT)
