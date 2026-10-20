# From fund documents to answers: a working sample on SEC filings

A work sample for a private-markets data science team. It takes documents an investor actually
receives, resolves every company name in them, pulls out the numbers, stores everything in one
queryable lake, and lets an analyst ask questions that come back with sources. Each step is
measured against an answer key, costed, and deployable with one command.

Built by Gurkaran Gulati with [Claude Code](https://claude.com/claude-code) as a pair programmer;
[`CLAUDE.md`](CLAUDE.md) explains how the work was divided.

## The problem

A secondaries team buys stakes in private funds. Before each bid, analysts read hundreds of
fund reports and filings about the companies inside those funds, and do the same four jobs
every time:

1. **Know what each company does.** Tag its sector consistently, and keep that model accurate
   as companies and language change from year to year.
2. **Know which company a name means.** Funds write "Actuant Corp" years after the company
   became Enerpac, abbreviate names, or misspell them.
3. **Get the numbers out.** Revenue and net income sit in long documents, sometimes in
   sentences instead of tables.
4. **Answer questions with sources.** For example, "Which of Fund III's holdings lost money
   last year?" A wrong number in a valuation is worse than "I don't know".

**The stand-in data, public and checkable:**
- 5,579 SEC 10-K filings from 2016–2020;
- the SEC's XBRL data as the answer key for numbers;
- the SEC's current and former company names;
- five synthetic fund reports (PDF) holding 54 positions in 50 real companies.

## What I built

![System map: sources flow through the classifier, name matcher and extraction graph into a Parquet data lake that the Q&A and lake agents query](docs/img/system_map.png)

| Piece | What it does | Folder |
|---|---|---|
| Sector classifier | Six ways to tag sectors, compared; a walk-forward backtest by year; deployed on AWS and Google Cloud | [`sectors/`](sectors) |
| Name matcher | Resolves a name through renames, typos and abbreviations, as of a date, and abstains when unsure; served as an API | [`matching/`](matching) |
| Extraction graph | A LangGraph flow that extracts, checks every number in code, retries, and sends failures to an analyst | [`extraction/`](extraction) |
| Document Q&A agent | A tool-using agent that answers questions about one filing, with every citation and number verified | [`extraction/qa_agent.py`](extraction/qa_agent.py) |
| Data lake | Fund PDFs + 10-Ks → resolved holdings, XBRL and extracted metrics, and text passages, as Parquet tables | [`lake/`](lake) |
| Lake analyst agent | Answers analyst questions over the lake with guarded SQL and citations; runs locally, or on Bedrock AgentCore with Athena | [`agent/`](agent) |

## How I approached it

- **Score honestly before comparing models.**
  - Split by company, never by filing; otherwise 46% of test filings would have their own
    company in training.
  - Choose settings on validation and open the test set once.
  - Write each decision rule down *before* the run that tests it.
- **Measure cost and latency next to quality.** Every row of every results table has a cost per
  1,000 items, a cost per correct answer, and latency.
- **Favor precision wherever a number or a company link lands.** The system abstains or routes
  to an analyst rather than guess. Retrieval favors recall; precision is enforced afterwards,
  with checks in code.
- **Treat every model output as untrusted.** Schemas, citations, numbers and SQL are all checked
  in code.
- **Make it reproducible.** Every stage is a `make` target, every run is logged, and every
  decision is written down with its alternatives.

## Key decisions

| Decision | Chosen | Over | Why |
|---|---|---|---|
| How to tag sectors | Few-shot LLM for new or changing labels; TF-IDF for bulk | Fine-tuning alone | Few-shot scored 0.782 with no training. TF-IDF is within 3 points at ~$0 |
| What goes in the prompt | 5 similar labeled examples | Hand-written rules | +4.9 points on validation. Examples carry the label convention in the companies' own words |
| Keeping the model current | Retrain yearly on all history | Freeze it | A frozen model loses 8 points on new companies by 2020 |
| Extraction | A fixed graph with checks and an analyst route | One LLM call | 71% vs. 0% of net income within 1% on unflagged filings; it knows when it doesn't know |
| Question answering | A tool-using agent with verification | One call over a fixed window | 91% vs. 72%, and right to say "not there" 95% vs. 57%, at 7× the cost |
| The lake | Parquet; DuckDB locally, S3 + Glue + Athena on AWS | Redshift, Postgres | $0 idle, no servers, the same files and SQL in both places |
| Where the agent runs | Bedrock AgentCore Runtime | Lambda, ECS | Built for long, multi-step agent sessions; deployed by CDK |
| How the agent reads data | Guarded text-to-SQL + retrieval | Pre-built queries only | Answers questions nobody planned for; the guard was tested against 61 attacks |

All twelve decision records, with evidence and "revisit when": [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Results at a glance

| Solution | Result | Compared with | Cost per 1,000 |
|---|---|---|---|
| Sector tagging (853 test filings) | **0.782** macro F1, few-shot, no training | TF-IDF 0.751 · fine-tuned 0.758 · zero-shot 0.748–0.781 | $2.71 (TF-IDF ~$0) |
| Next-year backtest | Retrained **0.73–0.75** on new companies | Frozen model falls to 0.65 by 2020 | — |
| Name matching (8,467 held-out names) | **98.8%** top-1 at 99% precision | Without former names, renamed companies drop to 13.6% | ~$0 |
| Extraction (119 filings vs. XBRL) | Net income within 1% on **71%** of unflagged filings; 24% to review | One LLM call: 0% | $7.86 |
| Document Q&A (91 questions) | **91%** correct; right to abstain 95% | One call: 72%; abstain 57% | $28.38 |
| Data lake (5 funds, 54 holdings) | 54/54 names resolved; 50 of 51 checks pass | Extracted vs. XBRL agree 82–84% | $0.46 per rebuild |
| Lake agent (57 labeled questions) | **95%** on AWS AgentCore, 93% locally; 100% on 5 of 7 question types | Cross-fund joins 50–67%: the weak spot | $13.31–14.84 |
| Deployment | Classifier live on AWS (SageMaker serverless) and Cloud Run; lake agent live on Bedrock AgentCore + Athena | Agent latency p50 11.5 s on AWS vs. 3.6 s locally (Athena per-query overhead) | $0 idle |

Tables with commentary: [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md). Every individual
run, with its log and per-item predictions: [`RESULTS.md`](RESULTS.md).

## What I learned

1. **How you score decides what you believe.**
   - A filing-level split would have flattered every model.
   - Returning companies score 0.94–0.98, but that is memory, not skill.
   - The backtest, not the test set, says how the model ages.
2. **Five labeled examples in a prompt beat fine-tuning here.** The signal is industry
   vocabulary. For millions of documents, TF-IDF at ~$0 is still the economical choice.
3. **LLM output must be checked in code.** Bedrock didn't enforce tool schemas, and one off-list
   answer cost 6.8 points of macro F1 through the scorer until I fixed both.
4. **Know what your data dropped.** The public corpus stripped every table, and that set the
   ceiling for extraction, not the model.
5. **Verification can be too strict or too loose, and only measurement tells you which:**
   - **Too strict:** the Q&A check cut coverage from 75% to 62% without raising precision.
   - **Too loose:** the lake agent's check passed an impossible "152.65% of NAV". The
     same question came back right (40.6%) on the next run, so plausibility checks are
     needed rather than hoping the model writes the SQL correctly.
6. **Thresholds tuned on easy data are too loose for money.** "Meta Platforms" matched SKYX
   Platforms at a score the tuned thresholds accept.
7. **Most deploy failures are permissions and quotas, not code:**
   - a SageMaker memory quota;
   - a Cloud Run build account without access;
   - Gemini prepaid credits;
   - a Bedrock use-case form.

## What I'd do better

| With more… | I would | Expected effect |
|---|---|---|
| **Time** | Measure retrieval on its own (recall@k against labeled passages) | Know whether a miss is retrieval's fault or the model's |
| | Add plausibility checks on agent answers; fix verification for derived numbers | Catch "152% of NAV"; Q&A coverage back to ~75% at the same precision |
| | Add an ambiguity rule and an adversarial name set for matching | Remove most of the remaining ~1% of wrong links |
| | Run the golden sets in CI on every change | Regressions blocked before they ship |
| **Data** | Real fund reports: scanned, varied layouts, hand-labeled | An honest number for messy documents |
| | About 100 filings per year for the extraction backtest; several quarters of fund reports | Drift you can actually see; point-in-time look-through |
| | GICS labels instead of SIC | Labels that match how investors think |
| **Money** | A larger model only on retries and cross-fund joins; two runs must agree on numbers | Precision up where errors cost most |
| | Bedrock Knowledge Bases + a re-ranker; provisioned concurrency | Better retrieval; no cold starts |
| **Different architecture** | Parse the filing's tables and inline XBRL; use the LLM only where no structure exists | Near-complete numbers at a fraction of the cost |
| | Event-driven ingestion (S3 → EventBridge → parse → lake) and Iceberg tables | New reports land automatically; point-in-time queries built in |
| | Bedrock Guardrails, real auth (IAM / Cognito) instead of an API key, prompt caching | Safer agents, proper access control, much lower agent cost |

Ranked, with how to prove each: [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md).

## Where to look

| Question | Read |
|---|---|
| Why each choice, and what it beat | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| The order things were tried, the bugs, and how each was caught | [`docs/ITERATIONS.md`](docs/ITERATIONS.md) |
| Every run: metrics, cost per correct answer, latency, logs, per-item predictions | [`RESULTS.md`](RESULTS.md) · [`results/all_runs.csv`](results/all_runs.csv) |
| Precision vs. recall for every task, measured, and how to flip each dial | [`docs/DIALS.md`](docs/DIALS.md) |
| Backtesting, cost, monitors and guardrails per solution | [`docs/OPERATIONS.md`](docs/OPERATIONS.md) |
| Architecture: AWS vs. Google Cloud, side by side | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| How each piece works, file by file | [`sectors/README.md`](sectors/README.md) · [`matching/WALKTHROUGH.md`](matching/WALKTHROUGH.md) · [`extraction/WALKTHROUGH.md`](extraction/WALKTHROUGH.md) · [`lake/WALKTHROUGH.md`](lake/WALKTHROUGH.md) · [`agent/WALKTHROUGH.md`](agent/WALKTHROUGH.md) |
| The documents and labels | [`lake/samples/`](lake/samples): fund report PDFs, truth labels, 57 golden questions |
| Every concept used, in plain words | [`docs/THEORY.md`](docs/THEORY.md) |
| Accounts, quotas, permissions, and every error hit | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) |

## Get started

**Requirements:** [uv](https://docs.astral.sh/uv/) (Python 3.14), Node.js (for the AWS CDK),
Docker (for the agent image), and the AWS and Google Cloud CLIs for the deploy steps. One-time
account setup is in [`docs/ONBOARDING.md`](docs/ONBOARDING.md).

```
git clone https://github.com/gurkarangulati/carlyle-alpinvest-interview && cd carlyle-alpinvest-interview
make setup                                                   # install everything
make help                                                    # list every stage

# data: public SEC filings and names (the SEC asks for a contact in the User-Agent)
make data names xbrl SEC_USER_AGENT="Your Name you@example.com"
make test

# experiments (Bedrock needs an AWS profile)
make all PROFILE=<aws-profile>                               # classifier, matching, extraction, Q&A
make lake PROFILE=<aws-profile> && make lake-validate        # the lake from fund PDFs + 10-Ks
make eval-agent PROFILE=<aws-profile>                        # the lake agent on 57 labeled questions
make results-index                                           # rebuild RESULTS.md from the runs

# deploy, check, tear down
make deploy-aws smoke-aws PROFILE=<aws-profile>              # classifier on SageMaker serverless
make deploy-gcp smoke-gcp PROJECT=<gcp-project-id>           # classifier on Cloud Run
make deploy-all PROFILE=<aws-profile>                        # lake (S3, Glue, Athena) + agent (AgentCore)
make ask-aws Q="Which funds hold CSX CORP?" PROFILE=<aws-profile>
make teardown-all teardown-aws PROFILE=<aws-profile>         # remove everything on AWS
make teardown-gcp PROJECT=<gcp-project-id>
```

`make -n <target>` shows what a target runs without running it. Browse the experiment history
with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. Data comes from
[EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) (Apache 2.0) and the
SEC's public APIs; nothing in `data/` is committed.

## Repo map

```
sectors/      classifier: split, baseline, CV, backtest, fine-tune, LLM; deploy/ (FastAPI, CDK, Cloud Run)
matching/     name matcher, golden set, evaluation, API
extraction/   extraction graph, document Q&A agent, evaluations
lake/         fund PDFs -> bronze / silver / gold Parquet, validation, golden questions, samples/
agent/        lake analyst agent, DuckDB / Athena backends, AgentCore app, deploy/ (CDK)
docs/         decisions, results summary, dials, operations, architecture, iterations, theory, onboarding
results/      every run's parameters and metrics (all_runs.csv), precision-coverage curves
tests/        220 tests
```

## Limits, stated plainly

- **Labels:** sector labels come from SIC codes; investors use GICS.
- **Data:** the corpus has no tables. The fund reports are synthetic and cleaner than real ones.
  Test sets are small, and differences under about 1.5 points are noise.
- **Production gaps:** the cloud stacks are demos, with API-key auth and no VPC or CI/CD.
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) lists what production needs.
