# Operating extraction and document Q&A

Two solutions share this folder: the **extraction graph** (a fixed LangGraph flow that pulls
revenue, net income and total assets from a 10-K) and the **document Q&A agent** (a tool-using
LangGraph agent that answers questions about one filing). Numbers come from `RESULTS.md`
§3–4 and `docs/DIALS.md`.

## 1. Backtesting

Neither solution is trained, so there is nothing to retrain. Backtesting here asks two
questions: **is it stable across filing years**, and **did a change make it worse**.

**Across years (measured, from saved predictions):**

| Fiscal year | Filings | Revenue within 1% | Net income within 1% | Sent to review | Q&A numeric accuracy |
|---|---|---|---|---|---|
| 2016 | 31 | 50% | 48% | 19% | 100% (14 questions) |
| 2017 | 22 | 47% | 48% | 41% | 77% (13) |
| 2018 | 22 | 17% | 53% | 32% | 83% (18) |
| 2019 | 25 | 67% | 73% | 8% | 100% (21) |
| 2020 | 19 | 8% | 67% | 21% | 90% (10) |

**Reading:** no steady decline, so there is no sign of drift. The swings are mostly small samples
(15–25 filings a year) and whether a filing states revenue in a sentence, since the corpus
stripped the tables.

**To make it a real backtest:**
- about 100 filings per year, scored per year;
- add filings newer than the prompt was written for, so new phrasing and new XBRL tags show up.

**Regression testing, the backtest of the *system*:** the 119-filing set and the 91 questions
are frozen golden sets. Every prompt, model or code change reruns them (`make extraction`,
`make qa`) and is compared with the last run in `RESULTS.md`. A change ships only if accuracy
on the not-flagged filings doesn't drop and the cost per correct answer doesn't rise.

## 2. Precision or recall

**Chosen: precision for numbers.** A wrong number in a valuation gets used; a blank gets noticed.

| Solution | What we do | Measured | To flip toward recall |
|---|---|---|---|
| Extraction | Failed checks → analyst review | Revenue precision 71% → 80%, no correct value lost; 24% reviewed | `MAX_RETRIES` 1 → 2; wider `RETRY_BUDGET`; loosen `MAX_RATIO` (`extraction/graph.py`, `checks.py`) |
| Q&A | Verify citations and numbers, abstain when not found | 91% correct; right to abstain 95%; verify cut coverage 75% → 62% with no precision gain | Accept numbers derived by `calculate` from seen inputs; raise `MAX_TOOL_CALLS` and `k` |
| Retrieval inside Q&A | 5 passages, TF-IDF | Not measured on its own yet | k 10–20 + hybrid search, **then** re-rank back to 5 to keep precision |

**To flip toward even more precision:** two runs must agree; the evidence quote must name the
fiscal year; show only verified, high-confidence Q&A answers (91.3%).

## 3. Cost

**Infrastructure: none.** Both run as Python on a laptop or a batch job. Bedrock is an API: you
need an IAM permission, and there's nothing to host or pay for while idle.

**Per use (measured, Claude Haiku 4.5 list price):**

| Solution | Per 1,000 | Per correct answer | Median latency |
|---|---|---|---|
| Extraction, single call (baseline) | $4.16 per 1,000 filings | n/a (0 correct) | 1.8 s |
| **Extraction graph** | **$7.86 per 1,000 filings** | ~$0.015 per correct net income | 3.4 s |
| Q&A, one call | $3.92 per 1,000 questions | $5.42 per 1,000 correct | 1.2 s |
| **Q&A agent** | **$28.38 per 1,000 questions** | $31.26 per 1,000 correct | 5.8 s (p95 17.9 s) |

**Cost levers:**
- **prompt caching** of the system prompt and tool results (the agent re-sends the conversation
  every turn);
- **Bedrock batch inference** at about half price for overnight runs;
- a larger model **only** on the retry.

## 4. Monitors

**In place:**
- Every run writes per-item rows: status, reason, retries, tokens, seconds and the full tool
  trace (`results/per_item/`).
- MLflow logs each run's accuracy, review rate, retry rate, cost and latency.
- `RESULTS.md` links every run to its console log.
- Live progress shows an ETA.

**Recommended in production, as CloudWatch metrics from the same fields:**

| Metric | Why | Alert when |
|---|---|---|
| Review rate | First sign of new document formats or a prompt regression | > 1.5× its 4-week average |
| Accuracy on a weekly golden sample (20 items) | Catches silent quality loss | Down more than 5 points |
| Cost per 1,000 and tokens per item | Prompt or loop growth | Up more than 25% |
| p95 latency, error and throttle rate | Capacity and quota | Errors > 1%; throttles sustained |
| Agent tool calls per question; hits on the 6-call cap | Looping | Cap hit on > 5% of questions |

## 5. Guardrails

**In place:**

| Risk | Guardrail |
|---|---|
| Model returns malformed or off-list output | Forced tool schema, then every field checked in code (Bedrock doesn't strictly enforce schemas) |
| Invented numbers | The evidence quote must appear in the filing; Q&A numbers must be in a cited passage or come from `calculate` |
| Invented inputs to a calculation | `verify` checks that every `calculate` input was seen in the filing |
| Code execution through the calculator | `ast` parsing; numbers and + − × ÷ only; never `eval` |
| Wrong scale ("in thousands") | Explicit scale field, applied in code; plausibility limits |
| Loops and runaway cost | 6 tool calls, 1 revision, 1 retry; a circuit breaker after 10 failures in a row; retry only on 429 and 5xx |
| Confident wrong answers | `needs_human_review` with a reason; "not in the document" is a correct answer |

**Recommended next:**
- **Bedrock Guardrails** on the agent: denied topics, PII redaction, and prompt-attack filters.
  Document text is data, and a filing could carry injected instructions.
- Per-run and per-day **spend budgets**.
- Keep the trace of every reviewed item for an audit trail.
