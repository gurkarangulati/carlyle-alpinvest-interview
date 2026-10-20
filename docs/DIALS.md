# Dials: precision vs. coverage, what we chose, and how to flip it

Every task here trades **coverage** (how often the system answers) against **precision** (how
often it's right when it does). Abstaining or sending an item to an analyst lowers coverage and
raises precision. The curves below are measured on saved per-item predictions by
`precision_curves.py` (`make precision-curves`), with no new model calls; the data is in
[`results/precision_coverage.csv`](../results/precision_coverage.csv).

## The principle: precision for numbers, recall at retrieval, precision at the answer

- **Numbers → precision.** A blank in a valuation model gets noticed and filled; a wrong number
  gets used. The system abstains or routes to an analyst unless a number is traced to its source.
- **Retrieval → recall first.** If the right passage isn't in the candidates, nothing later can
  recover it. Search casts a wide net (more passages, keyword plus semantic search). Precision is
  enforced *after* retrieval: re-ranking, code verification of citations and numbers, then
  abstaining.
- **Entity matching → precision.** A wrong company link silently corrupts every downstream join;
  "no match" goes to a short review queue.

## 1. Sector auto-tagging: confidence threshold

| Threshold | Fine-tuned: auto-tagged | Precision | TF-IDF: auto-tagged | Precision |
|---|---|---|---|---|
| none (tag everything) | 100% | 75.3% | 100% | 74.6% |
| 0.5 | 90% | 79.7% | 80% | 82.2% |
| 0.7 | 78% | 83.5% | 61% | 89.6% |
| **0.8 (chosen)** | **71%** | **86.1%** | 52% | 92.8% |
| 0.9 | 60% | 90.9% | 43% | 96.2% |
| 0.95 | 48% | 95.8% | 36% | 96.7% |

- **Chosen:** 0.8. Seven in ten filings are tagged automatically, and analysts see the rest.
- **To flip toward coverage:** set `THRESHOLD=0.5`, which tags 90% at about 80% precision. The
  `THRESHOLD` environment variable is in the Lambda in `sectors/deploy/aws/sector_stack.py` and in
  the Cloud Run app (`sectors/deploy/app/main.py`). Redeploy with `make deploy-aws` or
  `make deploy-gcp`, or change the Lambda's environment in place.
- **To flip toward precision:** `THRESHOLD=0.9` or `0.95`.
- **Note:** TF-IDF's probabilities are better *ranked* than the transformer's. At the same
  precision it tags fewer, but its top slice is cleaner.

## 2. Name matching: minimum score and margin (hybrid, dev split)

| Minimum score | Answers | Precision |
|---|---|---|
| 0–60 | 100% | 99.0% |
| **70 (chosen on dev)** | **99.9%** | **99.1%** |
| 90 | 97.0% | 99.3% |
| 95 | 89.3% | 99.3% |

- **Chosen:** the lowest thresholds that reach ≥98% precision on dev, applied to test once.
  Precision barely moves with the threshold: the remaining errors are **genuinely ambiguous
  names** (parent vs. subsidiary, two CIKs after a reorganization), not low scores.
- **To flip:** `uv run python -m matching.evaluate --split dev --target-precision 0.995` picks
  stricter thresholds and writes `matching/results/hybrid_thresholds.json`, which the test run
  and the API read. A lower `--target-precision` answers more.
- **The better lever** is an ambiguity rule: abstain when two or more companies share the
  normalized name and no `as_of` date separates them. See IMPROVEMENTS.
- **Warning from the lake build:** harder names fool these thresholds. As of 2019-12-31,
  "Meta Platforms" matched SKYX Platforms (score 71) and "Symantec" matched Stantec (score 80).
  Both cleared min score 70 and margin 0. The thresholds were tuned on a golden set that is easier
  than real names, so **for money decisions, tune on adversarial names**: near-miss companies,
  renames newer than the index, and short names. Raise `--target-precision` and require a margin
  over the runner-up.

## 3. Extraction: review routing

| Policy | Revenue: answered | Precision | Correct of all | Net income: answered | Precision |
|---|---|---|---|---|---|
| Accept every value | 55% | 71.1% | 39.0% | 66% | 86.3% |
| **Accept only values that passed the checks (chosen)** | 49% | **80.0%** | **39.0%** | 63% | 87.1% |

- **Chosen:** route failed checks to an analyst. For revenue, the checks removed **only wrong
  values**: precision went from 71% to 80%, and the number of correct values didn't change.
- **To flip toward coverage:**
  - raise `MAX_RETRIES` in `extraction/graph.py` (one more pass with a wider window, about 30%
    more cost);
  - widen `RETRY_BUDGET`;
  - loosen `MAX_RATIO` in `extraction/checks.py`.
- **To flip toward precision:** require the evidence quote to name the fiscal year, or accept a
  value only when two runs (or two models) agree.

## 4. Document Q&A: which answers to show (numeric questions)

| Policy | Answered | Precision |
|---|---|---|
| **Show every answer** | 75% | 89.5% |
| Only answers that passed verify | 62% | 89.4% |
| Verified + high confidence | 61% | 91.3% |

- **Finding:** the verify step **didn't buy precision** here. It sent correct answers to review
  (for example, growth computed from a number the model worked out in its head). **It's too strict
  on numbers**, which is the top Q&A improvement.
- **Chosen for now:** keep verify (it catches invented calculation inputs and bad citations), and
  fix its rule for derived numbers.
- **To flip toward recall:**
  - raise `MAX_TOOL_CALLS` (6) and the `search_filing` default `k`; `MAX_K` is 8 in
    `extraction/qa_tools.py`;
  - shrink `CHUNK` (1,500) with overlap.
- **To flip toward precision:** show only verified, high-confidence answers.

## 5. Retrieval (Q&A and the lake agent)

| Setting | Now | Toward recall | Toward precision |
|---|---|---|---|
| Passages returned (`k`) | 5 (max 8) | 10–20, then re-rank | 3 |
| Chunk size | 1,500 characters, no overlap | 800 with 200 overlap | larger chunks |
| Method | TF-IDF | + embeddings (hybrid, reciprocal rank fusion), query rewrites | + a re-ranker (e.g. Cohere Rerank on Bedrock) |
| Filters | company / fund | none | company + fiscal year + section |

**What's missing to tune it properly:** a retrieval-only metric, **recall@k** against labeled gold
passages per question. The lake's golden questions carry `expected_sources` for this.
