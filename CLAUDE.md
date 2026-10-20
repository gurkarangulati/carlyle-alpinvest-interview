# CLAUDE.md

Guidance for Claude Code working in this repository, and a record of how the repository was
built with it.

## What this repo is

A sandbox on public SEC 10-K filings (2016–2020) with three modules:

| Module | Question it answers |
|---|---|
| `sectors/` | Which of 10 sectors is a company in, judged from its 10-K business description? Does the answer hold up next year? Six ways to train or prompt are compared, followed by a walk-forward backtest and deployments on AWS and Google Cloud |
| `matching/` | Which company is this name, even after a rename, an abbreviation or a typo? Served behind an API |
| `extraction/` | What are the revenue, net income and total assets in this filing? Every number is checked against the SEC's XBRL data, and the flow is a LangGraph graph with retries and a human-review route |
| `extraction/qa_agent.py` | Answer a question about one filing, with citations. A tool-using agent searches the filing, reads the extracted numbers and calculates; code checks every citation and number |

## How it was built with Claude Code

- **Division of work.** The author chose the problems, set the rules below, ran every
  experiment from the terminal, predicted each result before running it, and decided every
  iteration from the numbers. Claude Code wrote the code and the docs, explained each part
  and its alternatives, and fixed errors as they came up.
- **Parallel modules.** `matching/` and `extraction/` were drafted in parallel by Claude Code
  subagents working from written specs. The main session reviewed them (tests, a scan for
  secrets, a read-through) before they were included.
- **The record.**
  - Every run's console output is in `sectors/results/logs/`.
  - Every run's settings and scores went to MLflow, exported to `sectors/results/results.csv`.
  - `docs/ITERATIONS.md` is the decision log, including the bugs and how each was caught.
- **Why say so.** The point of the repo is the method: honest splits, tuning on validation,
  backtesting, cost and latency next to accuracy, and production patterns. Using an AI
  coding assistant is part of that method, not something to hide.

## Layout

```
fixtures/make_companies.py   download EDGAR-CORPUS, fetch SIC codes, build data/companies.csv
sectors/                     classifier scripts (README.md is the runbook)
sectors/deploy/              FastAPI app, AWS CDK stack, Cloud Run script, teardown
sectors/results/             snapshot of run outputs (CSVs, console logs)
matching/                    name matching: fixture, golden set, matcher, evaluation, API
extraction/                  number extraction: XBRL fixture, LangGraph flow, evaluation
tests/                       pytest
docs/                        THEORY, ARCHITECTURE, ITERATIONS, ONBOARDING
data/                        git-ignored; rebuilt by the fixtures
```

## Commands (run from the repo root)

- Install everything: `uv sync`. Add `--group ml` for fine-tuning (torch, transformers, peft)
  and `--group deploy` for the CDK stack.
- Tests: `uv run pytest -q`
- Lint: `uv run ruff check .`
- Experiment tracking: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Every stage is a `make` target; `make help` lists them, and `make -n <target>` shows the commands.
- Each module's details are in its README or WALKTHROUGH.

## Rules that keep the scores honest

- **Split by company (cik), never by filing.** Otherwise the same company sits on both sides.
- **Choose on validation; open test once per model.** That covers prompt changes too. Write
  the decision rule down before seeing the result.
- **Never learn from test.** TF-IDF vocabulary and few-shot examples come from train only.
  The sector medians that extraction compares against come from train+val, never test.
- **The model never reads its own label.** The model never sees `sic_description` or the
  company name.
- **No future data in the backtest.** It trains only on years before the test year.
- **LLM output is untrusted.** Check every answer in code against the allowed values, the
  evidence quote and plausibility checks. Bedrock does not strictly enforce tool schemas.
- **Score only the real labels.** Pass `labels=`, so an off-list answer counts as a miss
  rather than creating an extra class.

## Conventions

- Plain scripts with argparse. Each starts with a docstring holding its run commands.
  Comments say *why*, not *what*.
- Each run prints a summary, saves predictions to CSV and logs to MLflow. Smoke runs
  (`--limit`) are never logged.
- LLM calls:
  - retry only 429/throttling and 5xx errors;
  - stop after 10 failures in a row;
  - report cost per 1,000 items and latency as well as accuracy.
- Services never log the document text.

## Secrets and accounts

Nothing secret lives in the repo. Code reads everything from the environment:

| Variable | Used for |
|---|---|
| `AWS_PROFILE` | Bedrock and the CDK deploy |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI |
| `GEMINI_API_KEY` | The Gemini API (only if you don't use Vertex) |
| `SEC_USER_AGENT` | SEC requests: "Name email", as the SEC requires |

`data/`, `mlflow.db`, model files and `.env` are git-ignored. Before committing, check
that nothing new contains a key, an account number or a local path.
