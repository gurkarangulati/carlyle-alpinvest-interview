# Results, with commentary

The headline tables for each solution, with what they mean. For every individual run (with its
log, cost and per-item predictions) see [`RESULTS.md`](../RESULTS.md).

## Sector tagging: 853 test filings, 10 sectors, split by company

| Approach | Test macro F1 | Accuracy | Training | Cost to score 1,000 filings |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.751 | 0.746 | 36 s on a laptop | ~$0 (2 ms) |
| Fine-tuned DistilRoBERTa, all 82M weights | 0.758 | 0.753 | 3.3 min | ~$0 on own hardware (2.5 s) |
| LoRA, 0.9% of weights | 0.744 | 0.734 | 1.4 min | same |
| Zero-shot Claude Haiku 4.5 (Bedrock) | 0.748 | 0.743 | none | $1.98 |
| Zero-shot Gemini 3.8 Flash (Vertex AI) | **0.781** | 0.770 | none | $1.52 |
| **Few-shot Claude Haiku 4.5, 5 retrieved training examples** | **0.782** | 0.770 | none | $2.71 |

- Prompt versions were compared on validation and the chosen one ran on test once. Validation:
  v1 0.756, rules 0.774, few-shot 0.804, rules + few-shot 0.814.
- The combination led by only 1.0, under the 1.5-point noise bar set before the run, so plain
  few-shot was tested.
- Macro F1 weights every sector equally.

## Backtest: does it hold up next year?

Walk-forward by year: train on every year before t, test on year t. F1 on companies the model has
never seen:

| Year | Retrained yearly | Frozen at 2016 | Last year only |
|---|---|---|---|
| 2017 | 0.685 | 0.685 | 0.685 |
| 2018 | 0.742 | 0.670 | 0.721 |
| 2019 | 0.755 | 0.693 | 0.708 |
| 2020 | 0.731 | 0.649 | 0.653 |

- Retrain yearly on all history: a frozen model is 8 points worse by 2020.
- Returning companies score 0.94–0.98, but that is memory, so quote the new-company number.

## Deployment (sector classifier)

| | AWS: API Gateway → Lambda → SageMaker | Google Cloud: Cloud Run |
|---|---|---|
| Cold start | 24.3 s | 4.3 s |
| Warm call | 0.53 s | 0.20 s |
| Idle cost | $0 | $0 |

At a 0.8 confidence threshold the model auto-tags 71% of filings at 86% accuracy; analysts see
the rest.

## Name matching: 8,467 held-out queries over 9,945 companies and 14,163 name aliases

| Method | Top-1 | Top-5 | Notes |
|---|---|---|---|
| Exact (normalized) | 0.716 | 0.719 | 0.001 on typos |
| Fuzzy (rapidfuzz) | 0.939 | 0.997 | |
| TF-IDF character n-grams | 0.968 | 0.988 | |
| TF-IDF without former names | 0.623 | 0.656 | **0.136 on former names** (vs. 0.995 with them) |
| **Hybrid + abstain** (thresholds from dev) | **0.988** | 0.994 | 99.0% precision when it answers; abstains 0.2% |
| Hybrid + Claude re-rank of close calls (dev) | 0.991 | 0.997 | +0.1 point at 21× the latency: not worth it |

The golden set is easier than real names. Names found in documents show real mislinks, and
harder probes fool the thresholds ("Meta Platforms" → SKYX Platforms).

## Extraction: 119 test filings, scored against SEC XBRL

| | Single LLM call | LangGraph flow (locate → extract → check → retry / review) |
|---|---|---|
| Revenue within 1% | 0% | 40% (55% on filings not flagged) |
| Net income within 1% | 0% | 57% (71% not flagged) |
| Sent to human review | 0% | 24%, mostly "no figures in the text" |
| Cost per 1,000 filings | $4.16 | $7.86 |

- The public corpus removed the HTML tables, so only numbers written in sentences survive.
- The single-call baseline reads the start of Item 8, which is usually the auditor's letter.

## Document Q&A: 91 questions on 40 filings

| | One call, no tools | Agent (search, financials, calculator) |
|---|---|---|
| Correct within 5%, counting a right "not in the document" | 72% | **91%** |
| Correct when it answered | 94% | 90% |
| Abstained correctly when the number isn't there | 57% | **95%** |
| Text questions answered (risks, customers, employees) | 20% | 93% |
| Citations valid | 100% | 100% |
| Cost per 1,000 questions / median latency | $3.92 / 1.2 s | $28.38 / 5.8 s |

## Data lake: 5 synthetic fund reports, 54 holdings, 50 real companies

| Check | Result |
|---|---|
| PDF parse vs. truth (holdings count, cost, fair value, NAV) | Exact for all 5 funds |
| Holdings matched to the right company | 54 / 54 |
| Extracted vs. XBRL within 1% | Revenue 82%, net income 84%; total assets 0% (tables stripped) |
| All checks | 50 pass, 1 warning, 0 hard fails; a clean rebuild reproduces them exactly |

## Lake analyst agent: 57 labeled questions

| Question type | Correct |
|---|---|
| Entity resolution, look-through, aggregation, financials, unanswerable | 100% (44 questions) |
| Text from risk factors | 86% (7) |
| Cross-cutting joins | 50% (6) |
| **Overall** | **93%** at $13.31 per 1,000 questions; 3.6 s median |

One cross-cutting answer reported "152.65% of NAV" and passed verification. Verify proves a number
came from a query, not that the query was right, so the next guardrail is a plausibility check on
the answer itself.
