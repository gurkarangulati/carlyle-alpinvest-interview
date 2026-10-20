# Decisions: what was chosen, over what, and why

One record per decision: the context, the options compared, the choice, the evidence (measured
here wherever possible; see `RESULTS.md`), and what would change the decision.

---

## 1. How to split and score

**Context.** Companies file every year, so their 10-Ks look alike from one year to the next.

| Option | For | Against |
|---|---|---|
| Random split by filing | Simple | 46% of test filings would have their own company in train, so the score measures memory |
| **Split by company, stratified by sector** | Honest, with balanced piles | One draw. Worst sector gap 1.5 points |
| StratifiedGroupKFold | Best balance (0.3 points), and gives a spread | 5× the training |

**Decision.** Split by company for model selection; use StratifiedGroupKFold to pick C and to
measure the noise (±1.4 points). Choose on validation, test once, and write the decision rule
down before each run.

**Revisit when** there are many more companies (one split becomes stable), or when the question
is about the future, which is what the walk-forward backtest answers.

## 2. Classic ML vs. fine-tuning vs. foundation models for tagging

| Option | Test macro F1 | Training | Cost per 1,000 | Latency | Operations |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.751 | 36 s | ~$0 | microseconds | A pickle |
| Fine-tuned DistilRoBERTa | 0.758 | 3.3 min | ~$0 on own compute | ms | A 332 MB model and an endpoint |
| LoRA | 0.744 | 1.4 min | same | ms | A 3 MB adapter per task |
| Zero-shot LLM (Gemini / Claude) | 0.781 / 0.748 | none | $1.52 / $1.98 | 3.4 s / 0.6 s | An API call |
| **Few-shot LLM (5 retrieved examples)** | **0.782** | none | $2.71 | 0.6 s | An API call plus a small index |

**Decision.** It depends on volume and how often the labels change:
- **Few-shot** for a new or changing taxonomy, or low volume: no training, and the best score.
- **TF-IDF or the fine-tuned model** for millions of documents, where cost is about zero.

The confidence threshold sends the uncertain items to analysts either way.

**Revisit when** labels move to GICS (rerun everything), or volume crosses ~1M documents a
month (LLM cost about $2.7k a month).

## 3. Rules vs. examples in the prompt

| Option | Validation macro F1 | Cost per 1,000 |
|---|---|---|
| Sector list (v1) | 0.756 | $1.98 |
| + Boundary rules from the SIC spec (v2) | 0.774 | $2.21 |
| **+ 5 similar labeled examples** | **0.804** | $2.70 |
| Rules + examples | 0.814 | $2.94 |

**Decision.** Plain few-shot. The combination won by 1.0 point, under the 1.5-point noise bar
set *before* the run. Examples carry the label convention in the companies' own words; rules
have to anticipate every case.

## 4. Where to serve the model

| Option | Idle cost | Cold start | Auth | Fit |
|---|---|---|---|---|
| **SageMaker serverless (AWS)** | $0 | 24 s measured | IAM; API key at the edge | Batch or occasional calls |
| SageMaker real-time | ~$80+ a month | none | IAM | Steady traffic, GPU, A/B variants |
| **Cloud Run (Google)** | $0 | 4.3 s measured | IAM identity token | Any container, simplest |
| Lambda with a container | $0 | seconds | IAM | Small models only |

**Decision.** Both serverless options, to compare them. The API key on AWS only meters usage;
production auth would be IAM, Cognito or a Lambda authorizer.

**Revisit when** users wait on answers: add provisioned concurrency or an always-on instance.

## 5. SageMaker vs. Bedrock vs. Vertex AI

**Your own model** (the fine-tuned classifier) goes on SageMaker or Cloud Run. **A provider's
foundation model** goes through an API: Bedrock (Claude, Llama, Mistral…) or Vertex AI
(Gemini). Bedrock's Custom Model Import takes some LLM architectures, not a RoBERTa classifier.

## 6. Name matching: what to trust

| Option | Test top-1 | Precision when answering | Cost |
|---|---|---|---|
| Exact after normalizing | 0.716 | n/a | ~0 |
| Fuzzy string score | 0.939 | n/a | ~0 |
| TF-IDF without former names | 0.623 (0.136 on renames) | n/a | ~0 |
| **Hybrid with aliases + abstain** | **0.988** | **0.990** | ~0 |
| Hybrid + LLM re-rank (dev) | 0.991 | 0.994 | 21× the latency, $1.13 per 1,000 calls |

**Decision.** The hybrid with former-name aliases, abstaining below thresholds tuned on dev.
The LLM re-rank isn't worth it on this golden set.

**Revisit when** names come from messy documents: the mentions demo shows mislinks that the
golden set doesn't. Hand-label ~100 real mentions and re-measure.

## 7. Extraction: one call vs. a graph

| Option | Net income within 1% (not flagged) | Review rate | Cost per 1,000 |
|---|---|---|---|
| One LLM call | 0% (it read the auditor's letter) | 0% | $4.16 |
| **LangGraph: locate → extract → check → retry → review** | 57% (71%) | 24% | $7.86 |
| Parse tables / read XBRL directly | n/a here (tables stripped) | n/a | ~0 |

**Decision.** The graph, because it knows when it doesn't know. Every review has a reason, and
most say the numbers aren't in the text.

**On real documents:** XBRL or table parsing first, with the LLM graph for documents that have
no structure (GP reports, letters).

## 8. Question answering: one call vs. an agent

| Option | Accuracy | Right to abstain | Cost per 1,000 | Median latency |
|---|---|---|---|---|
| One call with a fixed window | 72% | 57% | $3.92 | 1.2 s |
| **Tool-using agent + code verification** | **91%** | **95%** | $28.38 | 5.8 s |

**Decision.** The agent where a wrong number is costly (diligence, IC memos), and the cheap path
for bulk work.

**Guards:** a 6-call cap, a calculator that only does arithmetic, code checks on every citation
and number, and one revision before human review.

**First cost lever:** prompt caching, because each turn re-sends the conversation.

## 9. Fixed graph vs. free agent

A **fixed graph** (extraction) has the same steps every time: predictable cost, easy to test.
An **agent** (Q&A, lake analyst) chooses its steps: flexible, but needs caps, verification and
traces. Use a graph when the procedure is known, and an agent when the question decides the
procedure.

## 10. The lake: storage and query engine

| Option | Idle cost | Query cost | Operations | Fit |
|---|---|---|---|---|
| **Parquet on S3 + Glue catalog + Athena** | storage only | ~$5 per TB scanned | none | Occasional analyst queries over files |
| Redshift Serverless | $0 idle, but a minimum charge per query-second | higher | low | Heavy BI and joins at scale |
| Aurora / RDS Postgres | ~$40+ a month | included | patching, sizing | Transactional apps, pgvector |
| **DuckDB locally on the same Parquet** | $0 | $0 | none | Development and tests: same files, same SQL |

**Decision.** Parquet as the one format: DuckDB locally, Athena in AWS, and the agent switches
with one environment variable.

**Revisit when** query volume or concurrency grows (Redshift), or the app needs writes and
transactions (Aurora).

## 11. Where the agent runs

| Option | Idle cost | Session model | Ops | Notes |
|---|---|---|---|---|
| **Bedrock AgentCore Runtime** | pay per use | Long sessions, isolated per session, built for agents | Container in ECR + one resource | Pairs with AgentCore Memory, Gateway, Identity, Observability |
| Lambda | $0 | 15-minute limit, stateless | Simplest | Fine for short agents |
| ECS Fargate | Always-on tasks | Anything | Most to run | When you need full control |
| Cloud Run | $0 | Request-scoped | Simple | The Google equivalent |

**Decision.** AgentCore Runtime, deployed by CDK (the official `Runtime` construct). Agents run
long, multi-step sessions, and AgentCore is built for that.

## 12. How the agent reaches the data

| Option | For | Against |
|---|---|---|
| **Text-to-SQL with a guard** (one SELECT, allowlisted tables, LIMIT, sqlglot parse) | Answers questions nobody anticipated | The model writes SQL, which is the main risk, hence the guard and a read-only role |
| Pre-built query tools | Safe and predictable | Every new question needs code |
| **In-agent TF-IDF over `doc_chunks`** for text | No extra service, $0 | Misses paraphrases |
| Bedrock Knowledge Bases (S3 Vectors or OpenSearch) | Managed embeddings and retrieval | Per-query fees; OpenSearch has a high floor |

**Decision.** Guarded SQL plus in-agent retrieval for a demo-sized lake. Bedrock Knowledge Bases
on S3 Vectors is the production path once the corpus outgrows TF-IDF.
