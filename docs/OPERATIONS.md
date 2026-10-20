# Operations: backtesting, precision vs. recall, cost, monitors, guardrails

One row per solution. Each solution's own `OPERATIONS.md` has the detail. "Flip" means how to
move the dial the other way. Costs are measured where marked; infrastructure costs are list prices
to verify.

| Solution | Backtesting | Precision or recall (flip it) | Cost: idle / per use | Monitors in place | Guardrails in place | Detail |
|---|---|---|---|---|---|---|
| Sector classifier | **Walk-forward by year, done:** retrained 0.73–0.75 on new companies, frozen falls to 0.65 | Precision: auto-tag at ≥ 0.8 → 71% at 86% (flip: `THRESHOLD=0.5` → 90% at 80%) | $0 idle (serverless) / fractions of a cent per call; few-shot LLM $2.71 per 1k (measured) | Lambda metadata logs, SageMaker and API Gateway metrics, usage plan | Input limits, API key + throttle, confidence routing, no text in logs, Cloud Run IAM | [sectors/OPERATIONS.md](../sectors/OPERATIONS.md) |
| Name matching | Held-out test + point-in-time `as_of`; frozen-index test recommended | Precision: abstain below tuned thresholds, 99.1% (flip: `--target-precision`) | None / ~$0 (0.2 ms per name); LLM re-rank $1.13 per 1k calls | Eval by query type, per-query predictions | Abstain, `as_of`, input validation, LLM limited to the candidate list | [matching/OPERATIONS.md](../matching/OPERATIONS.md) |
| Extraction graph | **By fiscal year:** no steady decline (small samples); frozen golden set as a regression gate | Precision: failed checks → analyst; revenue 71% → 80% precise, no correct value lost (flip: more retries) | None / **$7.86 per 1k filings** (measured) | Per-filing status, reason, trace; MLflow; review rate | Schema checked in code, evidence quote, scale and plausibility, retry cap, circuit breaker | [extraction/OPERATIONS.md](../extraction/OPERATIONS.md) |
| Document Q&A agent | By fiscal year: 77–100% numeric accuracy; golden set as a regression gate | Precision at the answer, recall at retrieval: 91% correct, right to abstain 95% (flip: relax verify, raise `k`) | None / **$28.38 per 1k questions** (measured) | Full tool trace per question; MLflow | 6-call cap, arithmetic-only calculator, citation and number verification, abstain | [extraction/OPERATIONS.md](../extraction/OPERATIONS.md) |
| Data lake | Point in time by report date; reproducible rebuild (measured) | Precision: unresolved names stay NULL + abstain; failed extractions flagged, not stored as facts | S3 cents a month, Glue free, Athena $0 idle / $5 per TB scanned | `make lake-validate`: 50 checks, exits 1 on a hard fail | Parser never sees the truth files; totals must reconcile; private, encrypted, TLS-only bucket | [lake/OPERATIONS.md](../lake/OPERATIONS.md) |
| Lake analyst agent | 57-question labeled golden set as a regression gate; quarterly snapshots recommended | Precision: resolve names before filtering; numbers traced to tool results (flip: more tool calls, looser verify) | AgentCore $0 with no session, Athena per query / **$13.31 per 1k questions** (measured locally) | Per-answer trace; eval by type; AgentCore logs and traces | SQL guard (61 attacks tested), read-only engine, 1 GB scan cutoff, least-privilege IAM, no write path | [agent/OPERATIONS.md](../agent/OPERATIONS.md) |

**Recommended next across all:**
- Bedrock Guardrails (prompt-attack filter, PII) for the agents.
- Plausibility checks on agent answers (a share of NAV must be 0–100%).
- CloudWatch alarms on review rate and errors.
- A weekly golden-sample run with alerts.
- Real auth in place of the API key.
