# SEC 10-K sandbox: sector classification, company matching, number extraction

Four problems a private-markets data team meets every week, worked end to end on public
SEC filings (10-Ks, 2016–2020):

1. **Sectors** (`sectors/`): tag a company's sector from its own description of its
   business, and prove the model still works next year. Six approaches are compared: classic
   ML, fine-tuned transformers and foundation models (zero-shot, rules, few-shot). Then
   comes a walk-forward backtest, and the model is deployed as an API on AWS and on Google
   Cloud.
2. **Matching** (`matching/`): resolve a company name to the right entity through renames,
   abbreviations and typos. It abstains when unsure and is served behind an API.
3. **Extraction** (`extraction/`): pull revenue, net income and total assets out of a
   filing with an LLM inside a LangGraph flow. Every number is checked against the SEC's
   XBRL data, and the flow retries or sends the filing to a human when a check fails.
4. **Document Q&A** (`extraction/qa_agent.py`): ask a filing a question and get a cited
   answer from a tool-using agent. Code checks the citations and every number, and the agent
   says "not in the document" when the answer isn't there.

Built with [Claude Code](https://claude.com/claude-code); see `CLAUDE.md` for how the work
was divided.

## Results

### Sectors: 853 test filings, 10 sectors, split by company

| Approach | Test macro F1 | Accuracy | Training | Cost to score 1,000 filings |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.751 | 0.746 | 36 s on a laptop | ~$0 (2 ms) |
| Fine-tuned DistilRoBERTa, all 82M weights | 0.758 | 0.753 | 3.3 min | ~$0 on own hardware (2.5 s) |
| LoRA, 0.9% of weights | 0.744 | 0.734 | 1.4 min | same |
| Zero-shot Claude Haiku 4.5 (Bedrock) | 0.748 | 0.743 | none | $1.98 |
| Zero-shot Gemini 3.8 Flash (Vertex AI) | **0.781** | 0.770 | none | $1.52 |
| **Few-shot Claude Haiku 4.5, 5 retrieved training examples** | **0.782** | 0.770 | none | $2.71 |

Prompt versions were compared on validation, and the chosen one ran on test once. On
validation: v1 0.756, v2 with rules 0.774, few-shot 0.804, v2 + few-shot 0.814. The
combination was ahead by only 1.0, under the 1.5-point noise bar set before the run, so plain
few-shot was tested. Macro F1 weights every sector equally.

### Backtest: does it hold up next year?

Walk-forward by year: train on every year before t, then test on year t. F1 on companies the
model has never seen:

| Year | Retrained yearly | Frozen at 2016 | Last year only |
|---|---|---|---|
| 2017 | 0.685 | 0.685 | 0.685 |
| 2018 | 0.742 | 0.670 | 0.721 |
| 2019 | 0.755 | 0.693 | 0.708 |
| 2020 | 0.731 | 0.649 | 0.653 |

Retrain yearly on all history. A frozen model is 8 points worse by 2020. Returning companies
score 0.94–0.98, but that is memory, so quote the new-company number.

### Deployment

The fine-tuned model runs serverless on both clouds, at $0 while idle:

| | AWS: API Gateway → Lambda → SageMaker | Google Cloud: Cloud Run |
|---|---|---|
| Cold start | 24.3 s | 4.3 s |
| Warm call | 0.53 s | 0.20 s |

At a 0.8 confidence threshold the model auto-tags 71% of filings at 86% accuracy and sends
the rest to an analyst. See `docs/ARCHITECTURE.md`.

### Matching: 8,467 held-out queries over 9,945 companies and 14,163 name aliases

| Method | Top-1 | Top-5 | Notes |
|---|---|---|---|
| Exact (normalized) | 0.716 | 0.719 | 0.001 on typos |
| Fuzzy (rapidfuzz) | 0.939 | 0.997 | |
| TF-IDF character n-grams | 0.968 | 0.988 | |
| TF-IDF without former names | 0.623 | 0.656 | **0.136 on former names** (vs. 0.995 with them) |
| **Hybrid + abstain** (thresholds from dev) | **0.988** | 0.994 | 99.0% precision when it answers, abstains 0.2% |
| Hybrid + Claude re-rank of close calls (dev) | 0.991 | 0.997 | +0.1 point at 21× the latency: not worth it |

The golden set (SEC former names plus generated variants) is easier than names found in
documents. The mentions demo (`matching/results/mentions_demo.csv`) shows real mismatches.

### Extraction: 119 test filings, scored against SEC XBRL

| | Single LLM call | LangGraph flow (locate → extract → check → retry / review) |
|---|---|---|
| Revenue within 1% | 0% | 40% (55% on filings not flagged) |
| Net income within 1% | 0% | 57% (71% not flagged) |
| Sent to human review | 0% | 24%, mostly "no figures in the text" |
| Cost per 1,000 filings | $4.16 | $7.86 |

EDGAR-CORPUS removed the HTML tables, so only numbers written in sentences survive (the number
is in the text for about 57% of questions). Total assets lives in tables, so it is rarely
recoverable here. The single-call baseline reads the start of Item 8, which is usually the
auditor's letter; the Q&A baseline below is the fairer one-call comparison.

### Document Q&A: 91 questions on 40 filings

| | One call, no tools | Agent (search, financials, calculator) |
|---|---|---|
| Correct within 5%, counting a right "not in the document" | 72% | **91%** |
| Correct when it answered | 94% | 90% |
| Abstained correctly when the number isn't there | 57% | **95%** |
| Text questions answered (risks, customers, employees) | 20% | 93% |
| Citations valid | 100% | 100% |
| Cost per 1,000 questions / median latency | $3.92 / 1.2 s | $28.38 / 5.8 s |

The agent finds what one window of text misses, and it knows when the answer isn't there, at
7× the cost. Prompt caching is the first cost lever.

## Read in this order

1. `docs/THEORY.md`: every concept used, in plain words (leakage, splits, metrics, CV,
   backtesting, fine-tuning, LoRA, prompting, serving)
2. `docs/ITERATIONS.md`: every round, its numbers, and the decision it led to, including bugs
3. `sectors/README.md`: the runbook
4. `docs/ARCHITECTURE.md`: AWS vs. Google Cloud, side by side
5. `matching/WALKTHROUGH.md` and `extraction/WALKTHROUGH.md` (part 2 is the Q&A agent)
6. `docs/ONBOARDING.md`: accounts, quotas, permissions, and every error hit along the way

## Run it

Every stage is a `make` target (`make help` lists them). Settings are passed as variables or
read from the environment, and each target checks them before starting.

```
make setup                                              # uv sync --all-groups (Python 3.14)
make data names xbrl SEC_USER_AGENT="Your Name you@example.com"
make test
make all PROFILE=<aws-profile>                          # every experiment except the deploys
make llm-gemini PROJECT=<gcp-project-id>
make deploy-aws smoke-aws PROFILE=<aws-profile>         # API Gateway -> Lambda -> SageMaker
make deploy-gcp smoke-gcp PROJECT=<gcp-project-id>      # Cloud Run
make teardown-aws teardown-gcp PROFILE=... PROJECT=...  # then sectors/deploy/TEARDOWN.md
```

`make -n <target>` prints the commands without running them. Experiment tracking:
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Data comes from [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) (Apache
2.0) and the SEC's public APIs. Nothing in `data/` is committed. Cloud steps need your own
AWS and Google Cloud accounts (`docs/ONBOARDING.md`).

## Weak points

- **The labels are SIC codes mapped to 10 sectors.** Investors use GICS, and some "errors"
  (oilfield equipment as energy) are right under GICS.
- **The transformer read only the first 256 tokens.** It needs 512 tokens or chunking for
  a fair contest with TF-IDF's 4,000 characters.
- **The test set is small:** 853 filings, with 32–34 in the smallest sectors. Differences
  under about 1.5 points are noise.
- **EDGAR-CORPUS stripped the HTML tables,** so extraction works from numbers in sentences.
  On real filings, parse the tables or the XBRL directly.
- **The cloud stacks are demos:** API-key auth, no VPC, no CI/CD. `docs/ARCHITECTURE.md`
  lists what production needs.
