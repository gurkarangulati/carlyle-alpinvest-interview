# Operating name matching

## 1. Backtesting

- **Held out:** thresholds are tuned on dev companies and scored once on test companies (98.8%
  top-1 at 99% precision).
- **Point in time:** `as_of` uses only names valid on that date. "Walt Disney Co" returns the
  old entity as of 2015 and the new one as of 2020.
- **Over time (recommended):** freeze the alias index at date T, then score names that changed
  *after* T. That measures how quickly an index goes stale, the matching version of the
  classifier's frozen arm.

## 2. Precision or recall

**Chosen: precision.** A wrong link corrupts every join; an abstain lands in a short review queue.
Thresholds give the most coverage at ≥ 98% precision on dev.

**Measured:** raising the minimum score to 95 costs 11% coverage for +0.3 points of precision.
The remaining errors are ambiguous names, not low scores.

**To flip:**
- `uv run python -m matching.evaluate --split dev --target-precision <x>`. A lower value answers
  more; a higher one abstains more.
- Better: an ambiguity rule, plus tuning on adversarial names ("Meta Platforms" vs. SKYX
  Platforms).

## 3. Cost

- **Infrastructure:** none required. It runs as a Python library or a local FastAPI app. On
  Cloud Run or Lambda it would cost $0 while idle.
- **Per use:** about 0.2 ms per name, effectively $0.
- **The optional LLM re-rank** costs $0.26 for 231 close calls ($1.13 per 1,000 calls) at 21×
  the latency. It isn't used by default.

## 4. Monitors

**In place:**
- Evaluation by query type: top-1, top-5, MRR, abstain rate and precision when answering.
- Per-query predictions.
- MLflow runs.

**Recommended:**
- The daily abstain rate (rising means new names the index doesn't know).
- Agreement with analyst corrections from the review queue.
- Monthly refresh of SEC names, with the alias count per company.

## 5. Guardrails

**In place:**
- Abstain thresholds and `as_of`.
- The API validates input length, and its error responses never echo the input.
- The LLM re-rank may only pick from the candidate list or "none": the answer is checked in
  code, with a budget cap and a circuit breaker.

**Recommended:**
- An ambiguity rule.
- A human-confirmed alias table that overrides the model.
- An audit log of every match used in a valuation.
