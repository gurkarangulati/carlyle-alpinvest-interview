# Sectors: tag a company's sector from its 10-K, and check it holds up next year

Run every command from the repo root. On a new machine or cloud account, do
`docs/ONBOARDING.md` first; it covers logins, quotas and permissions.

## Data (once)

| Step | Command | Writes |
|---|---|---|
| Download EDGAR-CORPUS 2016–2020 (test + validate) | `uv run python fixtures/make_companies.py download` | `data/raw/edgar/` (~1.8 GB) |
| Keep each filing's business section | `uv run python fixtures/make_companies.py texts` | `data/companies_text.jsonl` |
| SIC code per company, from the SEC | `SEC_USER_AGENT="Your Name you@example.com" uv run python fixtures/make_companies.py sic` | `data/sic_cache.json` |
| Map SIC → 10 sectors, one row per filing | `uv run python fixtures/make_companies.py merge` | `data/companies.csv` (5,579 filings, 3,988 companies) |

## Runbook

| # | Step | Command | Writes |
|---|---|---|---|
| 1 | Split by company, 70/15/15, stratified by sector, seed 14 | `uv run python sectors/split.py` | `data/sector_split.json` |
| 1b | Compare 4 split methods over 20 seeds | `uv run python sectors/split_options.py` | `out/split_options.csv` |
| 2 | Baseline: TF-IDF + logistic regression, C chosen on val, test once | `uv run python sectors/baseline.py` | `out/tfidf_logreg_*.csv` |
| 3 | Wider C, then balanced class weights | `uv run python sectors/baseline.py --c 10 30 100 300 1000 [--balanced]` | `out/tfidf_logreg[_balanced]_*.csv` |
| 3b | Choose C by 5-fold CV, grouped by company | `uv run python sectors/cv.py --c 10 30 100 300 1000` | `out/cv_folds.csv` |
| 4 | Walk-forward backtest by year: retrained, frozen and last-year arms | `uv run python sectors/backtest.py --c 100` | `out/backtest.csv` |
| 4b | Fine-tune DistilRoBERTa: all weights, or LoRA (needs `uv sync --group ml`) | `uv run python sectors/finetune.py [--lora]` | `out/models/`, `out/finetune_*_test_preds.csv` |
| 4c | Foundation model: zero-shot, rules (v2), or few-shot | `uv run python sectors/llm_classify.py --provider bedrock\|vertex\|gemini --pile val\|test [--guide v2] [--shots 5] [--limit 30]` | `out/llm_classify_*_preds.csv` |
| 5 | One results table from MLflow | `uv run python sectors/report.py` | `out/results.csv` |
| 6a | Package the fine-tuned model | `uv run python sectors/deploy/package_model.py` | `out/model.tar.gz` |
| 6b | Serve locally | `cd sectors/deploy/app && MODEL_DIR=../../out/models/finetune_full uv run uvicorn main:app --port 8080` | http://localhost:8080/docs |
| 6c | Deploy to AWS (API Gateway → Lambda → SageMaker serverless) | `cd sectors/deploy/aws && npx aws-cdk@latest deploy` | a live API |
| 6d | Deploy to Google Cloud (Cloud Run) | `bash sectors/deploy/gcp/deploy.sh` | a live private service |
| 6e | Tear down | `sectors/deploy/TEARDOWN.md` | |

`out/` is git-ignored. `results/` is a committed snapshot of `out/` from the runs recorded
in `docs/ITERATIONS.md`, including every console log.

For `llm_classify.py`, tune prompts on `--pile val` and run `--pile test` only for the version
chosen there. The `--limit` flag is for smoke tests and is never logged.

## Files

| File | What it does |
|---|---|
| `common.py` | Paths, `load_piles()`, MLflow setup, `score()`, and the log tee that saves every run's console output |
| `split.py` | Company-level split |
| `split_options.py` | How evenly four split methods spread the sectors, over 20 seeds |
| `baseline.py` | TF-IDF + logistic regression, C sweep on val, per-sector report, confusion matrix, top words |
| `cv.py` | GridSearchCV with StratifiedGroupKFold (TF-IDF inside the pipeline, so no fold leaks) |
| `backtest.py` | Walk-forward by year, scored separately on companies seen in training and new ones |
| `finetune.py` | Hugging Face Trainer: full fine-tune or LoRA (peft) |
| `llm_classify.py` | Gemini (API or Vertex) or Claude on Bedrock, with structured output checked in code; zero-shot, rules or retrieval few-shot |
| `report.py` | Every MLflow run in one table |
| `deploy/` | FastAPI app, Dockerfile, CDK stack (Python), Lambda handler, Cloud Run script |
