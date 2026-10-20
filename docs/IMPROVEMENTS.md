# Improvements: what would make each part better, ranked by expected value

Each item has the expected effect, its cost, and the measurement that would prove it. Numbers
cited are from `RESULTS.md` and `docs/DIALS.md`.

## Extraction (numbers)

| # | Change | Expected effect | Cost | Prove it with |
|---|---|---|---|---|
| 1 | **Read the real tables:** parse the filing's HTML or inline XBRL instead of the table-stripped corpus | Total assets from about 11% to near full coverage; revenue and net income up sharply | Engineering, no LLM cost | Same 119 filings, same scorer |
| 2 | Look in the right places: Item 6 (Selected Financial Data, which exists for 2016–2020) and "Results of Operations" in Item 7 | More values found on the first pass | ~0 | Retry rate falls; coverage rises |
| 3 | Escalate only the retry to a larger model | Fixes the hard cases without paying for them on easy ones | ~20–30% more per run | Accuracy on flagged filings |
| 4 | Two runs (or two models) must agree before a value is accepted | Precision up; some coverage down | 2× | The precision–coverage table |
| 5 | Year check on the evidence: the quote must name the fiscal year | Fewer prior-year values read by mistake | ~0 | Errors where the year was wrong |
| 6 | Explicit loss rule: "net loss of $4.2 million" becomes −4.2M | Fewer sign errors | ~0 | `sign_errors` |
| 7 | Labeled truth for documents without XBRL (GP reports) | Measurement where it matters most | Analyst hours | A new golden set |

## Retrieval and document Q&A

| # | Change | Expected effect | Cost | Prove it with |
|---|---|---|---|---|
| 1 | **Fix verify for derived numbers:** let a number pass if it is the output of a `calculate` call whose inputs were all seen in tool results | Coverage from 62% back toward 75% at the same ~89% precision | ~0 | The policy table in DIALS §4 |
| 2 | Measure retrieval on its own: recall@k against gold passages | Shows whether a miss is retrieval's fault or the model's | Labeling ~50 questions | recall@5, recall@10 |
| 3 | Hybrid retrieval: TF-IDF/BM25 + embeddings (Titan or Cohere via Bedrock), fused | Catches paraphrases ("headcount" vs. "employees") | Embedding cost, small | recall@k |
| 4 | Re-rank the top 20 down to 5 | Precision back after widening recall | Per-query re-rank fee | Accuracy and citation precision |
| 5 | Smaller chunks with overlap (800 / 200) | Numbers stay with their sentence | More chunks | recall@k |
| 6 | Prompt caching of the system prompt and tool results | Cost down a lot, because each turn re-sends the conversation | ~0 | Cost per 1,000 questions |
| 7 | Metadata filters (company, fiscal year, section) before search | Fewer wrong-year passages | ~0 | Errors where the year was wrong |

## Name matching

| # | Change | Expected effect | Prove it with |
|---|---|---|---|
| 1 | **Ambiguity rule:** abstain when two or more companies share the normalized name and no date separates them | Removes most of the remaining ~1% of errors | Precision when answered |
| 2 | Hand-label ~100 names found in real documents | An honest number for messy text (the golden set is easier) | The mentions precision |
| 3 | Acronym and short-name dictionary (IBM, P&G) | Recall on short names | Top-1 by query type |
| 4 | Use ticker, state and address when two candidates tie | Fewer subsidiary mix-ups | Precision when answered |

## Sector tagging

| # | Change | Expected effect |
|---|---|---|
| 1 | Read the whole Item 1 (512 tokens or chunks) for the transformer | A fair contest with TF-IDF's 4,000 characters |
| 2 | Relabel by GICS instead of SIC | Labels match how investors think; some "errors" become right |
| 3 | Combine TF-IDF and few-shot: agree → auto-tag, disagree → analyst | 76% agree at 87% accuracy (measured) |
| 4 | Per-sector thresholds | Fewer analyst reviews for the easy sectors |
