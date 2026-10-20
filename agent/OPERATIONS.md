# Operating the lake analyst agent

A LangGraph agent answers analyst questions over the lake: funds, holdings, metrics and filing
text. It runs locally on DuckDB, or on AWS on Bedrock AgentCore Runtime with Athena. See
`WALKTHROUGH.md` for how it works and `deploy/TEARDOWN.md` for removal.

## 1. Backtesting

The agent isn't trained, so its "backtest" is **regression against a frozen, labeled golden
set**: 57 questions whose answers come from the truth files and XBRL, never from the lake.
- Every change to the prompt, model, tools or lake reruns `make eval-agent` (local) or
  `make eval-agent TARGET=aws`, and is compared by question type.
- **Smoke result so far:** 10/10 across all types; citations 100% valid; 0% reviewed; 1.6 tool
  calls per question.
- **Over time:** rebuild the lake each quarter with that quarter's fund reports, and keep each
  quarter's golden set. The agent is then scored on every past quarter too (point in time:
  questions say "as of" a date).

## 2. Precision or recall

**Chosen:** precision at the answer, and recall at retrieval and lookup.
- **Names:** `resolve_company` abstains below its thresholds, and the agent must resolve a
  name before filtering on it. An unresolved name returns "not in the lake", never a guess.
- **Numbers:** `verify` requires every number in the answer to appear in a tool result or a
  `calculate` output. Otherwise there's one revision, then `needs_human_review`.
- **SQL:** `LIMIT 200` and an allowlist favor safe, small answers over completeness.

**Known limit:** verify proves a number came from a query, not that the query asked the right
thing. A wrong filter gives a "verified" wrong answer, and only the golden set catches it.

**To flip toward recall:**
- raise the matcher's coverage (lower `--target-precision`);
- raise `MAX_TOOL_CALLS`;
- return more passages from `search_docs`;
- let `verify` accept numbers derived by `calculate` chains.

**To flip toward even more precision:**
- show only answers with status `done` and high confidence;
- require a SQL citation for every number;
- add a second model as a checker for aggregation questions.

## 3. Cost

**Infrastructure (AWS, `make deploy-all`):**

| Piece | Idle | Per use |
|---|---|---|
| S3 (lake Parquet, ~tens of MB) | cents a month ($0.023 per GB-month) | requests are fractions of a cent |
| Glue Data Catalog (1 database, 8 tables) | $0 (first million objects free) | — |
| Athena workgroup | $0 | $5 per TB scanned, 10 MB minimum per query; this lake is MBs, so cents per thousand queries |
| AgentCore Runtime | $0 when no session is active | per second of CPU and memory while a session runs (check current pricing) |
| ECR image (CDK assets repo) | cents a month | — |
| CloudWatch logs and traces | cents at demo volume | per GB ingested |

**Per question (measured locally, Claude Haiku 4.5):** about $10 per 1,000 questions; p50
latency 4.8 s, p90 5.8 s. On AWS, add Athena time (seconds) and the first-session cold start: the
matcher and 8,490 chunks load through Athena on the first request.

**Cost levers:** prompt caching (3–4k tokens are re-sent each turn); preload the chunk index
into the image or onto S3 as one file.

## 4. Monitors

**In place:**
- Every answer carries a trace: every LLM turn, tool call, SQL query id and verify result.
- `evaluate.py` writes accuracy by type, citation validity, review rate, tool calls, cost and
  latency to `results/` and to MLflow.
- AgentCore's CDK construct grants CloudWatch logs, X-Ray tracing and `PutMetricData`, so the
  runtime emits logs and traces.

**Recommended:**
- CloudWatch alarms on errors, p95 latency and Athena bytes scanned.
- A dashboard of review rate and tool calls per question.
- A weekly golden-sample run, with an alert on a drop of more than 5 points.
- The same metrics as extraction (see `extraction/OPERATIONS.md`).

## 5. Guardrails

**In place:**

| Risk | Guardrail |
|---|---|
| Model writes destructive or exfiltrating SQL | `guard_sql` (sqlglot): one SELECT only; allowlisted lake tables; no other schemas, `information_schema`, table functions, file paths or engine settings; CTE scope checked; LIMIT 200. **61 attack cases tested** |
| The engine does something the guard missed | DuckDB attached read-only with external access off and a locked configuration; Athena workgroup with a 1 GB scan cutoff and enforced settings; 30 s query timeout |
| Over-broad cloud permissions | One IAM role: `InvokeModel` on the Haiku profile only, Athena on this workgroup, Glue read on this database, S3 read on `lake/*`, write only on `athena-results/*` |
| Invented numbers or sources | `verify`: citations produced in this run, numbers traced to tool outputs |
| Loops and cost | 6 tool calls, 1 revision, retries only on throttling and 5xx |
| Wrong company | Resolve before filtering; abstain when unsure |
| Data exposure | No write path anywhere; logs hold traces, not the documents' full text |

**Recommended next:**
- **Bedrock Guardrails**: prompt-attack filter, denied topics, and PII redaction. Fund
  documents are data, and could carry injected instructions.
- AgentCore Identity or an authorizer in front of the runtime, so access is per user.
- Per-user rate limits and spend budgets.
