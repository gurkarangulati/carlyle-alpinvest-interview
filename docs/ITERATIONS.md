# Iterations: every round, what it showed, and what was decided

In run order. "Val" is validation macro F1 and "test" is test macro F1 (853 filings).
Console logs for each run are in `sectors/results/logs/`.

## Classifier

| # | Round | Result | Decision |
|---|---|---|---|
| 1 | Split by company, 70/15/15, stratified by sector, seed 14 | 2,791 / 598 / 599 companies; filings 69.5 / 15.2 / 15.3%; worst sector gap 1.7 pts | Keep. A filing-level split would have put 46% of test filings next to their company in train |
| 2 | Baseline: TF-IDF + logistic regression, C from 0.3 to 30 | C = 30: val 0.733, test 0.751, accuracy 0.746. Trains in 36 s; scores 1,000 filings in 2 ms | The bar every other model must clear |
| 3 | Wider C, 10 to 1,000 | C = 100: val 0.741, test 0.751 | Flat on test: more C buys nothing |
| 4 | Balanced class weights | Val 0.758, test 0.750. Communications recall 0.44 → 0.47, precision 0.93 → 0.88 | Kept plain: better on val, nothing on test |
| 5 | 5-fold CV, StratifiedGroupKFold | C 30 / 100 / 300 = 0.729 / 0.730 / 0.732 ± 0.014; test 0.749 at C = 300 | The C differences are noise. CV is the steadier way to choose |
| 6 | Split methods over 20 seeds | Worst sector gap: random 3.9, by sector 1.5, by sector + filing count 0.9, StratifiedGroupKFold 0.3 | Keep the stratified split; use StratifiedGroupKFold in production |
| 7 | Walk-forward backtest, 2017–2020, C = 100 | New-company F1: retrained 0.685 / 0.742 / 0.755 / 0.731; frozen 0.685 / 0.670 / 0.693 / 0.649 | Retrain yearly on all history. Quote about 0.73–0.75 for new companies |
| 7b | Last-year-only arm (same size as frozen) | Overall gain over frozen: +4.8, +2.0, 0.0 | Most of the lift is more data, not just fresher data |
| 8 | Fine-tune DistilRoBERTa, all 82M weights, 3 epochs | Val 0.758, test 0.758, accuracy 0.753. 3.3 min; 2.5 s per 1,000 filings; 332 MB. Val loss rose at epoch 3 | A tie with TF-IDF at about 1,000× the scoring cost. It read 256 tokens vs. TF-IDF's 4,000 characters |
| 9 | LoRA, rank 8, query/value, 0.9% of weights | Val 0.766, test 0.744. 1.4 min; 3 MB adapter | Nearly as good with a tiny trainable footprint |
| 10 | Confidence routing on the fine-tuned model | At 0.8: 71% auto-tagged at 86% accuracy. At 0.9: 60% at 91% | The threshold is a business dial |
| 11 | Zero-shot Claude Haiku 4.5 (Bedrock), test | First scored 0.680, then **0.748** after a fix (see bugs) | Matches TF-IDF with zero labeled data |
| 12 | Zero-shot Gemini 3.8 Flash (Vertex), test | **0.781**, accuracy 0.770; $1.52 per 1,000; median 3.4 s, p95 23 s | Best single model so far. It thinks, so it's slow |
| 13 | Claude as a second opinion on TF-IDF's low-confidence half | Right 56.5% vs. TF-IDF's 54.8% | Not worth $2 per 1,000 as a fixer. Agreement between the two (76% of filings, 87% accurate) is a confidence signal |
| 14 | Prompt v1 vs. v2 (boundary rules from the SIC spec), val, Claude | 0.756 → 0.774 | Just over the 1.5-point noise bar |
| 15 | Few-shot: 5 nearest **train** filings with labels, val, Claude | **0.804** (+4.9 over v1) | Examples teach the label convention better than rules |
| 16 | v2 rules + 5 shots, val | 0.814 (+1.0 over few-shot) | Under the 1.5-point bar set before the run, so plain few-shot goes to test |
| 17 | Few-shot, test, once | **0.782**, accuracy 0.770; $2.71 per 1,000; median 0.63 s | Ties Gemini zero-shot (0.781) at a fifth of the latency |

## Deployment

| # | Step | What happened | Fix / lesson |
|---|---|---|---|
| D1 | FastAPI locally | Energy text → energy 0.94 auto_tag; short text → 422 | Validate input before the model |
| D2 | `cdk synth` | CDK added an account-wide `ApiGateway::Account` resource | `cloud_watch_role=False`: a stack shouldn't change account settings |
| D3 | `cdk deploy` #1 | SageMaker refused 4,096 MB: the account limit is 3,072 MB per serverless endpoint | Sized to 3 GB (model is 332 MB). The failed first create rolled back cleanly |
| D4 | `cdk deploy` #2 | 17 resources up. Cold call 24.3 s, warm 0.53 s | Cold start is close to API Gateway's 29 s limit |
| D5 | `gcloud run deploy --source` #1 | PERMISSION_DENIED: the build service account couldn't read the source | Granted `roles/run.builder` |
| D6 | Cloud Run #2 | Up. First call 4.3 s, warm 0.20 s; private behind an IAM identity token | The model is inside the image, so there's no download at start |

## Bugs and how they were caught

| Bug | How it showed | Fix |
|---|---|---|
| The log tee lacked `isatty`, so transformers crashed | Traceback at Trainer start | `__getattr__` passes attributes through to the screen |
| transformers 5.20 renamed `warmup_ratio` | TypeError | `warmup_steps=0.1` |
| Gemini free tier: 429 on every call; the script retried all 853 | A wall of 429s | Retry only 429/5xx, with longer backoff; stop after 10 failures in a row |
| AI Studio key billed prepaid credits (402) | Every call refused | Switched to Vertex AI (Cloud Billing, IAM) |
| **One off-list LLM answer ("software") cost 6.8 points of macro F1** | The per-class report listed an 11th class | `score()` passes `labels=`; answers are checked in code, and off-list ones count as `invalid` |
| A run that fails mostly would still be logged | A design review | No MLflow logging if more than 5% of calls fail |
| Vertex run looked stuck (20 minutes with no output) | No progress line | Lesson: batch jobs print progress and an ETA. Vertex batch prediction for large runs |

## Matching

| # | Round | Result | Decision |
|---|---|---|---|
| M1 | Golden set: SEC former names + generated variants, split dev/test by company | 16,992 queries (dev 8,525, test 8,467) | Tune on dev, score test once |
| M2 | Five methods on dev | Hybrid 0.990 top-1; TF-IDF without aliases 0.613 (0.132 on former names) | Aliases are the single biggest win |
| M3 | Abstain thresholds on dev (most coverage at ≥98% precision) | min score 70, min margin 0: 99.1% precision, 99.9% coverage | Saved to `results/hybrid_thresholds.json`, read by test |
| M4 | Test, once | Hybrid 0.988 top-1, 99.0% precision | Holds up on new companies |
| M5 | Claude re-rank of 231 close calls (dev) | 0.990 → 0.991; 0 invalid answers; $0.26 | Not worth 21× the latency on this set |
| M6 | Mentions in 20 business sections | 43 found, 81% linked; caught renames (Actuant → Enerpac) and made wrong links (Heartland → Usio) | Real text is harder than the golden set; hand-label ~100 mentions next |

## Extraction and document Q&A

| # | Round | Result | Decision |
|---|---|---|---|
| E1 | Inspected the corpus | HTML tables stripped; the statements' numbers survive only in sentences | Score within 1% and 5%; expect low total-assets coverage |
| E2 | Matched filings to XBRL (no accession in the corpus) | 94% matched by fiscal year + 10-K + original filing | Rule and failure cases documented in `fetch_xbrl.py` |
| E3 | Tool schema returned nested fields as text | Bedrock doesn't enforce schemas | Flat keys, and check every field in code |
| E4 | Single call vs. graph, 119 filings | Revenue within 1%: 0% vs. 40% (55% not flagged); review 24% | The graph's locate + retry + review route earns its 1.9× cost |
| E5 | Q&A agent copied numbers from the prompt's example into `calculate` | Smoke test | Removed the example; `verify` now checks calculation inputs come from the filing |
| E6 | Q&A, 91 questions: one call vs. agent | 72% vs. 91%; correct abstention 57% vs. 95% | Use the agent where a wrong number is costly; it costs 7× more |
