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
5. **Data lake + analyst agent** (`lake/`, `agent/`): fund-report PDFs and 10-Ks become
   queryable tables. Every reported name is resolved to one company, and metrics are extracted
   and reconciled with XBRL. A LangGraph agent answers analyst questions over those tables with
   guarded SQL, retrieval and citations, locally on DuckDB or on Bedrock AgentCore with Athena.
   It is scored on 57 labeled questions.

Built with [Claude Code](https://claude.com/claude-code); see `CLAUDE.md` for how the work
was divided.

**Where to look**
- [`RESULTS.md`](RESULTS.md): every run of every task, with quality, cost per 1,000, **cost per
  correct answer**, latency, and links to its console log and per-item predictions
- [`results/all_runs.csv`](results/all_runs.csv): the full history, every parameter and metric of every run
- [`docs/DECISIONS.md`](docs/DECISIONS.md): each architecture and modeling choice, the options it beat, and the evidence
- [`docs/ITERATIONS.md`](docs/ITERATIONS.md): the order things were tried, the bugs, and what each round decided
- [`docs/DIALS.md`](docs/DIALS.md): precision vs. coverage for every task, measured, with how to flip each dial
- [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md): what would make each part better, ranked, with how to prove it

## System map

![System map: sources flow through the classifier, name matcher and extraction graph into a Parquet data lake that the Q&A and lake agents query](docs/img/system_map.png)

Read left to right. The matcher turns every reported name into one company (the entity master),
extraction turns text into numbers checked against XBRL, and both land in the lake that the agents
query with cited answers. Foundation models are API calls (Bedrock, Vertex AI); nothing hosts them.

## What is built, what ran, what is deployed

| Component | Built | Ran and measured | Deployed |
|---|---|---|---|
| Sector classifier, 6 approaches | ✅ `sectors/` | ✅ 17 runs, including the backtest ([RESULTS §1](RESULTS.md)) | ✅ AWS (API Gateway → Lambda → SageMaker serverless, CDK) and ✅ Google Cloud Run |
| Name matcher + API | ✅ `matching/` | ✅ dev, test and LLM re-rank ([RESULTS §2](RESULTS.md)) | Local API (`uvicorn matching.api:app`) |
| Extraction graph | ✅ `extraction/` | ✅ 119 filings vs. XBRL ([RESULTS §3](RESULTS.md)) | Local |
| Document Q&A agent | ✅ `extraction/qa_agent.py` | ✅ 91 questions ([RESULTS §4](RESULTS.md)) | Local |
| Fund-report PDFs + data lake | ✅ `lake/` ([sample PDFs](lake/samples/fund_reports)) | ✅ 50 checks pass, 1 warning; holdings matched 54/54 ([validate.csv](lake/results/validate.csv)) | `make deploy-lake`: S3 + Glue + Athena |
| Lake analyst agent | ✅ `agent/` | ✅ Local smoke test 10/10; full 57-question golden set via `make eval-agent` | `make deploy-agent`: Bedrock AgentCore Runtime |
| Foundation models | Called through APIs | Bedrock Claude Haiku 4.5, Vertex AI Gemini 3.8 Flash | No infrastructure: an IAM permission and an API call |

## Operating each solution: backtesting, precision vs. recall, cost, monitors, guardrails

Summary only; each solution's `OPERATIONS.md` has the detail. "Flip" means how to move the dial the
other way. Costs are measured where marked; infrastructure costs are list prices to verify.

| Solution | Backtesting | Precision or recall (flip it) | Infrastructure cost: idle / per use | Monitors in place | Guardrails in place | Detail |
|---|---|---|---|---|---|---|
| Sector classifier | **Walk-forward by year, done:** retrained 0.73–0.75 on new companies, frozen falls to 0.65 | Precision: auto-tag at ≥ 0.8 → 71% at 86% (flip: `THRESHOLD=0.5` → 90% at 80%) | $0 idle (serverless) / fractions of a cent per call; few-shot LLM $2.71 per 1k (measured) | Lambda metadata logs, SageMaker and API Gateway metrics, usage plan | Input limits, API key + throttle, confidence routing, no text in logs, Cloud Run IAM | [sectors/OPERATIONS.md](sectors/OPERATIONS.md) |
| Name matching | Held-out test + point-in-time `as_of`; frozen-index test recommended | Precision: abstain below tuned thresholds, 99.1% (flip: `--target-precision`) | None / ~$0 (0.2 ms per name); LLM re-rank $1.13 per 1k calls | Eval by query type, per-query predictions | Abstain, `as_of`, input validation, LLM limited to the candidate list | [matching/OPERATIONS.md](matching/OPERATIONS.md) |
| Extraction graph | **By fiscal year:** no steady decline (small samples); frozen golden set as a regression gate | Precision: failed checks → analyst; revenue 71% → 80% precise, no correct value lost (flip: more retries) | None / **$7.86 per 1k filings** (measured) | Per-filing status, reason, trace; MLflow; review rate | Schema checked in code, evidence quote, scale and plausibility, retry cap, circuit breaker | [extraction/OPERATIONS.md](extraction/OPERATIONS.md) |
| Document Q&A agent | By fiscal year: 77–100% numeric accuracy; golden set as a regression gate | Precision at the answer, recall at retrieval: 91% correct, right to abstain 95% (flip: relax verify, raise `k`) | None / **$28.38 per 1k questions** (measured) | Full tool trace per question; MLflow | 6-call cap, arithmetic-only calculator, citation and number verification, abstain | [extraction/OPERATIONS.md](extraction/OPERATIONS.md) |
| Data lake | Point in time by report date; reproducible rebuild (measured) | Precision: unresolved names stay NULL + abstain; failed extractions flagged, not stored as facts | S3 cents/month, Glue free, Athena $0 idle / $5 per TB scanned | `make lake-validate`: 50 checks, exits 1 on a hard fail | Parser never sees truth; totals must reconcile; private, encrypted, TLS-only bucket | [lake/OPERATIONS.md](lake/OPERATIONS.md) |
| Lake analyst agent | 57-question labeled golden set as a regression gate; quarterly snapshots recommended | Precision: resolve names before filtering; numbers traced to tool results (flip: more tool calls, looser verify) | AgentCore $0 with no session, Athena per query / ~$10 per 1k questions (measured locally) | Per-answer trace; eval by type; AgentCore logs and traces | SQL guard (61 attacks tested), read-only engine, 1 GB scan cutoff, least-privilege IAM, no write path | [agent/OPERATIONS.md](agent/OPERATIONS.md) |

**Recommended next across all:** Bedrock Guardrails (prompt-attack filter, PII) for the agents,
CloudWatch alarms on review rate and errors, a weekly golden-sample run with alerts, and real auth
in place of the API key.

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

1. [`docs/DECISIONS.md`](docs/DECISIONS.md) and [`RESULTS.md`](RESULTS.md): the choices and the numbers behind them
2. `docs/THEORY.md`: every concept used, in plain words (leakage, splits, metrics, CV,
   backtesting, fine-tuning, LoRA, prompting, serving)
3. `docs/ITERATIONS.md`: every round, its numbers, and the decision it led to, including bugs
4. `sectors/README.md`: the runbook
5. `docs/ARCHITECTURE.md`: AWS vs. Google Cloud, side by side
6. `matching/WALKTHROUGH.md` and `extraction/WALKTHROUGH.md` (part 2 is the Q&A agent)
7. `docs/ONBOARDING.md`: accounts, quotas, permissions, and every error hit along the way

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

`make -n <target>` prints the commands without running them. `make results-index` rebuilds `RESULTS.md` after new runs. Experiment tracking:
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
