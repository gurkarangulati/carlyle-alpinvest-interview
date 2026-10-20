# Company-name matching: a walkthrough

This module answers one question: **"Which company is this name?"**
It takes a messy name, like `Walt Disney Company` or `Apple Hospitaltiy REIT`,
and returns the SEC company (its CIK) that the name refers to. If it is not sure,
it says so. This task is called *entity resolution* or *record linkage*.

---

## 1. The problem, and why a secondaries team cares

A secondaries fund buys stakes in existing private-equity funds. Before it buys,
it has to understand what is inside each fund: the portfolio companies.

That information comes from many places:

- quarterly reports from the general partners (GPs),
- capital account statements,
- SEC filings, news, and data vendors.

The same company shows up under different names in each one:

- "Acme Holdings, Inc." in one GP report, "Acme Hldgs" in another.
- "Alabama Gas Corp" in an old report, "Spire Alabama Inc" today (it was renamed).
- A ticker in one place, a legal name in another.

If you cannot tell these are the same company, you double count exposure, or miss it.
Two funds may both hold the same company, and that concentration is a real risk.
So matching names to one ID is the first step before any portfolio analysis.

Here we practise on public data: SEC filers. The SEC gives each filer a permanent ID
(the CIK), and it records every name a company has used. That gives us right answers
to test against.

---

## 2. The data

| File | What it is | Made by |
|---|---|---|
| `data/companies.csv` | One row per 10-K filing: cik, year, name, sector, business text. 3,988 companies, 2016–2020. | earlier project |
| `data/sec_names.jsonl` | One line per company in companies.csv: current name, **former names with dates**, tickers. | `fetch_sec_names.py` |
| `data/company_tickers.json` | SEC's list of every listed company (10,443 tickers, 8,029 companies), current name and ticker only. | `fetch_sec_names.py` |
| `data/match_golden.csv` | Test queries with the right CIK. | `golden.py` |

**The universe** is both SEC files merged by CIK: 9,945 companies and 14,163 names (current plus former). The ticker list has 8,029 companies; 2,072 of our 3,988 are in it (the rest have since delisted, merged or gone private).
Why add the ticker list? With only our 3,988 companies, a matcher would rarely meet
a look-alike. The bigger list adds real competition: other banks, other "Apple"s.

---

## 3. The golden set (the answer key)

`golden.py` makes queries where we already know the right CIK.

| Type | Example query | Right company | Why it is here |
|---|---|---|---|
| `former_name` | `ALABAMA GAS CORP` | Spire Alabama Inc | **The main test.** A real rename, from SEC's own records. |
| `former_variant` | `ALEXANDR & BALDWIN INC` | Alexander & Baldwin | An old name, also with an edit. |
| `suffix` | `SPIRE ALABAMA Corp` | Spire Alabama Inc | Inc → Corp, Corp → Corporation, or dropped. |
| `the` | `The KROGER CO` | Kroger Co | SEC drops a leading "The"; reports often keep it. |
| `ampersand` | `Procter and Gamble Co` | Procter & Gamble Co | `&` ↔ `and` |
| `abbreviation` | `DIXIE Grp INC` | Dixie Group Inc | International ↔ Intl, Holdings ↔ Hldgs, ... |
| `case_punct` | `Aar Corp` | AAR CORP | Case and punctuation changes. |
| `typo` | `ABBOTT LABBORATORIES` | Abbott Laboratories | One wrong letter: swapped, missing or doubled. |
| `ticker` | `ABT` | Abbott Laboratories | Tickers. |

Rules that keep the test honest:

- **Former names that differ only in case or tags are skipped.** `AIR PRODUCTS & CHEMICALS INC /DE/` → `Air Products & Chemicals, Inc.` is not a real rename.
- **Every company gets a typo, plus one other edit.** Rare edits (ampersand, abbreviation) are picked first when they apply, so they have enough queries to measure.
- **Split by company, not by query.** All queries about one CIK go to dev or all go to test (seed 14, 50/50). If they were split by query, a rule tuned on "Acme Corp" in dev would get a free pass on "Acme Inc" in test.
- **Dev is for tuning, test is opened once.** The hybrid thresholds are chosen on dev and saved. The test run reads them; it cannot change them.
- **`n_owners` column:** how many companies share the query's cleaned-up name. Above 1 means even a perfect matcher cannot be sure from the name alone.

Counts from the build (your rerun gives the same numbers, since the seed is fixed):

| type | dev | test | all |
|---|---|---|---|
| `abbreviation` | 342 | 350 | 692 |
| `ampersand` | 48 | 58 | 106 |
| `case_punct` | 496 | 543 | 1039 |
| `former_name` | 1764 | 1744 | 3508 |
| `former_variant` | 1764 | 1744 | 3508 |
| `suffix` | 516 | 501 | 1017 |
| `the` | 592 | 542 | 1134 |
| `ticker` | 1041 | 1025 | 2066 |
| `typo` | 1962 | 1960 | 3922 |
| **all** | **8,525** | **8,467** | **16,992** |

That is 16,992 queries about all 3,988 companies. 2,467 companies have at least one former name (4,218 in all; 3,508 remain after dropping case-only changes). 145 queries have a cleaned-up name shared by 2+ companies.

---

## 4. The code, file by file

### `common.py`: shared paths and loaders
- Paths to every data file, so no script hard-codes one.
- `load_universe()` merges the two SEC files into one record per CIK.
- `start_mlflow()` points MLflow at `mlflow.db` in the repo root, experiment "matching".

### `fetch_sec_names.py`: download names from the SEC
- Reads the User-Agent from the `SEC_USER_AGENT` environment variable. The SEC blocks
  requests without a contact. Reading it from the environment keeps names and emails
  out of the code. If it is missing, the script stops with a clear message.
- **Rate limit:** one call every 0.13 s at most (under 8 per second; the SEC allows 10).
- **Retries:** 429, 5xx and dropped connections wait 1, 2, 4, 8 seconds and try again. A 404 means
  "no such company" and is not retried. Other 4xx errors (like 403, a bad User-Agent)
  stop the script, because retrying will not fix them.
- **Resumable:** each company is written to disk the moment it arrives. On restart,
  companies already in the file are skipped.
- Keeps only five fields per company. The full SEC file has every filing; we need names.

### `matcher.py`: the core
**Part 1: `normalize()`** turns a name into a comparable key:
1. lowercase;
2. remove SEC tags like `/DE/` (state of incorporation);
3. `&` → `and`;
4. remove `.` and `'` with no gap (`U.S.` → `us`, `Macy's` → `macys`), other symbols become spaces;
5. swap known short forms to one spelling (`intl` → `international`, `incorporated` → `inc`);
6. drop a leading `the`;
7. drop legal suffixes at the end (`inc`, `corp`, `co`, `llc`, `plc`, ...).

So `The Walt Disney Company` and `WALT DISNEY CO/` both become `walt disney`.

**Part 2: the alias index.** Every company gets one row per name it has used:
its current name and each former name, with the dates each was valid.
The current name is valid from the day the last former name stopped.
Tickers sit in a separate exact lookup table.

**Part 3: the methods.**

| Method | How it works | Good at | Bad at |
|---|---|---|---|
| `exact` | Cleaned query must equal a cleaned alias. | Fast, never fuzzy-wrong. | Any typo. |
| `fuzzy` | rapidfuzz `token_set_ratio` against every alias. | Word order, extra words. | Gives 100 to "apple" vs "apple hospitality reit" (subset of words). Slow on big lists. |
| `tfidf` | Split names into 3–5 letter pieces (char n-grams), weight rare pieces higher (TF-IDF), find the closest name by cosine similarity. | Typos, spacing. Fast. | Short names, which have few pieces. |
| `tfidf_noalias` | Same, with former names hidden. | Shows what aliases are worth. | Every rename. |
| `hybrid` | TF-IDF picks 20 companies. Each is re-scored with rapidfuzz. **Abstains** when the best score is low or the runner-up is too close. | Best of both, and knows when it is unsure. | Thresholds need tuning. |

The hybrid re-score is the average of `token_set_ratio` (forgives missing words) and
`token_sort_ratio` (punishes them). The average forgives a little, not completely.

**Part 4: `decide()`, the abstain rule.**
Answer only if both hold:
- best score ≥ `min_score` (something looks right), and
- best score − runner-up score ≥ `min_margin` (it is clearly ahead).

A small margin means two companies look equally right. Guessing there is a coin flip,
and a wrong link is worse than a "please check" for a portfolio team.

**Part 5: tickers.** A query that looks like a ticker (`AAPL`, `BRK-B`) is looked up
exactly in the ticker table. Codes are never fuzzy-matched: `AAPL` and `APLE` are
different companies.

**Part 6: `as_of`, point-in-time matching.** Pass a date, and only names valid on that
date are used. `Walt Disney Co` in 2015 meant the company now called TWDC Enterprises 18.

### `golden.py`: builds the answer key
- `make_variant(name, kind, rng)` makes one edit of one kind, or returns `None`
  if the edit does not apply (no `&` to swap, for example).
- `typo()` changes one letter in a word of 4+ letters, never the first letter.
- A seeded random generator (`random.Random(14)`) makes the file the same every run.

### `evaluate.py`: scores every method
- Runs all five methods on the same queries.
- **top1:** the first answer is right. **top5:** the right company is anywhere in the
  first 5 (recall@5). **MRR:** 1 for rank 1, ½ for rank 2, ⅓ for rank 3 ... 0 if missing.
- **Hybrid only:** `precision_answered` (right when it answers) and `abstain_rate`.
  For the hybrid, top1 counts an abstain as a miss.
- **On dev,** it tries a grid of `min_score` × `min_margin` and keeps the pair that answers
  the most queries while precision stays ≥ 98% (`--target-precision`). It saves them to
  `results/hybrid_thresholds.json`. **On test,** it only reads that file.
- `secs_per_1k_queries`: wall time per 1,000 queries on this machine. For `hybrid_llm`
  it is the LLM pass alone (on top of the hybrid time), spread over all queries.
- Saves the summary to `results/eval_<split>.csv` and every prediction to
  `out/preds_<split>.csv` (git-ignored). Logs to MLflow, except `--limit` smoke runs.

### `llm_rerank.py`: optional LLM tie-breaker
- Only for **low-margin** queries: the top two hybrid scores are within 5 points and the
  best is at least 70. Everything else never touches the LLM. That keeps cost tiny.
- Calls Claude Haiku 4.5 on Amazon Bedrock with `converse` and a **forced tool**
  `pick_company`, whose input is one of the candidate CIKs or `"none"`.
- **The answer is checked in code.** Bedrock does not strictly enforce a tool's schema,
  and an off-list answer has happened before. Anything not in the candidate list counts
  as `invalid` and becomes an abstain.
- **Retries** only throttling (429) and server errors (5xx): wait 2, 4, 8 seconds.
- **Circuit breaker:** after 10 failures in a row it stops calling. A permission or
  quota error will not fix itself, so there is no point paying for 1,000 failures.
- **Budget:** stops calling when spend reaches `--llm-budget` (default $1.50).
- Uses the AWS profile from the `AWS_PROFILE` environment variable. Nothing is hard-coded.

### `mentions.py`: names inside documents
- **Find:** a regex for runs of Capitalized Words that end in a legal suffix
  (Inc, Corp, Corporation, LLC, Ltd, plc, Co., Holdings, L.P.). Small joining words are
  allowed: `Bank of America Corporation`, `Procter & Gamble Co.`
- Leading heading or sentence words are trimmed: `BUSINESS General Acme Inc` → `Acme Inc`.
- **Link:** each mention goes through the hybrid matcher.
- **Demo only, not scored.** There are no labels for mentions. The printout includes a weak
  sanity check: in how many documents did we find and link the filer's own name (most
  Item 1 sections name the filer in the first lines)?
- **How to score it properly:** hand-label about 100 mentions from 20 documents. For each,
  record whether the span is right and which CIK it is (or "not in universe"). Then report
  the finder's precision and recall and the linker's accuracy separately, so you know
  which half to fix.

### `api.py`: the HTTP service (FastAPI)
- The index is built **once** at startup (FastAPI `lifespan`), not per request.
- `GET /match?name=...&k=5&as_of=YYYY-MM-DD` → ranked candidates plus `match` or `abstain`.
- `POST /mentions` with `{"text": ...}` → each mention and its match.
- `GET /health` → status and index size.
- **Input limits:** name 1–200 characters, k 1–20, text up to 20,000 characters, dates
  must parse. Bad input gets a 422 that says what was wrong **without echoing the input**.
- **No logging of text.** Run with `--no-access-log`, because uvicorn's access log would
  print every query string.

### `tests/test_matching.py`
Unit tests with a tiny made-up universe (no network, no data files): `normalize()`
examples, each variant kind, the typo rule, the abstain rule, former names, `as_of`,
ticker lookup and the mention regex.

---

## 5. Options considered, and why these were chosen

| Decision | Options | Chosen, and why |
|---|---|---|
| Ground truth | Hand-label pairs; buy a vendor mapping; use SEC former names | **SEC former names.** Real renames, with dates, free, and thousands of them. Hand labels are slow; vendor data is not public. |
| Candidate universe | Only our 3,988 companies; all SEC filers | **Ours plus the ~10,400 listed filers.** Bigger haystack, more look-alikes, a fairer test. Fetching all ~800k SEC filers would be slow and mostly shell companies. |
| Legal suffixes | Keep and standardize; drop | **Drop at the end of the name.** "Acme Inc" vs "Acme Corp" is almost always the same business in a report. Risk: two real companies that differ only by suffix. |
| Fuzzy scorer | `ratio`, `token_sort_ratio`, `token_set_ratio`, `WRatio` | **`token_set_ratio`** as the plain baseline (the brief asked for it); the **average of set and sort** in the hybrid, because set alone ties "apple" with "apple hospitality reit". `WRatio` mixes several scorers with hidden weights, which is harder to explain. |
| Vector method | Word TF-IDF; char n-gram TF-IDF; sentence embeddings | **Char 3–5-grams.** Typos break whole words but leave most n-grams intact. Embeddings know meaning, but company names are mostly proper nouns, and embeddings need a model and more compute. Worth testing next. |
| Two stages | Fuzzy over everything; TF-IDF only; TF-IDF then fuzzy | **TF-IDF then fuzzy (retrieve, then re-rank).** TF-IDF is one sparse matrix multiply; rapidfuzz on 20 candidates is cheap. This is the same shape as search engines and RAG. |
| Unsure cases | Always answer; abstain; ask an LLM | **Abstain, with an optional LLM tie-break.** A wrong link silently corrupts exposure numbers; an abstain goes to a human queue. |
| LLM output | Free text; JSON in the prompt; forced tool with an enum | **Forced tool + check in code.** The tool makes the shape predictable; the code check catches the rare off-list answer. |
| Thresholds | Hand-picked; tuned on dev | **Grid search on dev, frozen for test.** Picking on test would overstate results. |

---

## 6. Weaknesses and failure modes

- **Common names.** "First National Bank", "American Software". Many companies share the
  same cleaned name. The matcher should abstain (margin near 0) and does, but it cannot
  pick the right one without more context (state, industry, address).
- **Parents vs. subsidiaries.** "Alabama Power Co" and "Southern Co" are different CIKs.
  A GP report may name the parent when it means the subsidiary, or the reverse. Names alone
  cannot settle this; you need an ownership table.
- **Mergers and reorganizations with two CIKs.** In 2019 the old Walt Disney Co was renamed
  TWDC Enterprises 18 Corp., and a new holding company took the name "Walt Disney Co" under
  a new CIK. The query "Walt Disney Co" matches both at score 100, so the hybrid abstains.
  With a date it resolves: `--as-of 2015-06-30` gives the old CIK (1001039),
  `--as-of 2020-06-30` the new one (1744489). Without a date, no name-only method can know.
- **Dates are partial.** Companies that come only from the ticker list have no former names
  and no start date, so `as_of` treats their current name as always valid. Example: Apple's
  former name "Apple Computer" is missing, because Apple is not in our 10-K set.
- **Abbreviations it does not know.** The synonym table is short and hand-written.
  "IBM" for International Business Machines, "P&G", "BofA", "GE" all fail unless the query is
  also the ticker. Fix: an acronym rule (first letters of each word) or an alias table.
- **Legal-suffix dropping can merge two companies** that differ only by suffix.
- **Short names.** A 3-letter name has few n-grams, so TF-IDF scores are noisy.
- **Ticker look-alikes.** An all-caps one-word name like `NVIDIA` looks like a ticker. It is
  only routed to the ticker table if it IS a ticker; otherwise it falls back to names.
  Tickers are also reused over time, and our table holds only today's tickers.
- **The regex mention finder misses:**
  - names without a legal suffix ("Google", "our partner Walmart");
  - names that start with a lowercase letter ("eBay Inc.", "iRobot Corp");
  - names that run into the words before them ("The President of Acme Inc." → the span
    includes "President of"); headings are trimmed only for a short list of words;
  - "Company" as a suffix is left out on purpose, because "the Company" means the filer.
- **The golden set is easier than real life.** Variants are generated by rules, so they look
  like the rules. Real GP reports have messier errors (OCR, nicknames, fund-specific codes).
  Former names are exact SEC strings, so any method with aliases finds them easily; the
  `former_variant` and `typo` rows are the harder test.

---

## 7. Run it, in order

All commands run from the repo root.

```bash
# 1. Fetch SEC names (about 10-15 minutes, resumable; rerun the same command if stopped)
SEC_USER_AGENT="Your Name your@email" uv run python -m matching.fetch_sec_names
#    expect: "saved .../company_tickers.json (10443 listed filers)", progress every 250,
#            then "done: 3988 CIKs cached"

# 2. Build the golden set
uv run python -m matching.golden
#    expect: the counts table in section 3, then example queries

# 3. Unit tests
uv run python -m pytest tests/test_matching.py -q
#    expect: all tests pass

# 4. Smoke test (a few queries per type, not logged)
uv run python -m matching.evaluate --split dev --limit 20

# 5. Full dev run: compares methods and chooses the hybrid thresholds
uv run python -m matching.evaluate --split dev
#    expect: a top-1 table by type and method, an overall table, and
#            results/eval_dev.csv + results/hybrid_thresholds.json

# 6. Optional: the LLM tie-break on dev (needs Bedrock access; spend is printed)
AWS_PROFILE=<your-profile> uv run python -m matching.evaluate --split dev --llm-rerank

# 7. Test, once, after you are done tuning
uv run python -m matching.evaluate --split test
#    (add --llm-rerank with AWS_PROFILE=... to score the LLM variant on test too)

# 8. Mentions demo (after step 5, so it uses the thresholds chosen on dev)
uv run python -m matching.mentions
#    expect: mentions found, % linked, filer self-check, results/mentions_demo.csv

# 9. Try single names
uv run python -m matching.matcher "Walt Disney Company"                  # abstains: two CIKs
uv run python -m matching.matcher "Walt Disney Co" --as-of 2015-06-30   # old CIK 1001039
uv run python -m matching.matcher "Walt Disney Co" --as-of 2020-06-30   # new CIK 1744489

# 10. The API
uv run uvicorn matching.api:app --port 8000 --no-access-log
curl "http://127.0.0.1:8000/match?name=Spire%20Alabama&k=3"
curl -X POST http://127.0.0.1:8000/mentions -H "Content-Type: application/json" \
     -d '{"text": "We compete with Global Payments Inc. and Acme Widgets, Inc."}'

# 11. Look at the runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

What to look for in step 5:
- `tfidf_noalias` on `former_name` should be far below `tfidf`: that gap is the value of
  keeping former names.
- `exact` should be near 0 on `typo` and near perfect on `suffix`, `the`, `case_punct`:
  that shows what `normalize()` alone buys.
- The hybrid's `precision_answered` vs `abstain_rate` is the trade-off a portfolio team
  would choose: how many "please check" items they accept to keep wrong links rare.

---

## 8. Interview questions, with short answers

**1. Why not just use fuzzy matching on everything?**
It is slow (every query against every name), and its scores do not say when it is unsure.
TF-IDF narrows the field cheaply; fuzzy re-scores 20 names; the margin tells us when to stop.

**2. How did you build ground truth without labelling by hand?**
SEC records every former name of a company, with dates. A former name is a real query
with a known answer. We added rule-based variants and tickers for the other kinds of mess.

**3. Why split by company?**
Queries about one company are related. If "Acme Corp" is in dev and "Acme Inc" in test,
anything tuned on dev has already seen the test company. Splitting by CIK keeps test unseen.

**4. How did you choose the abstain thresholds?**
A grid of score and margin on dev. We keep the pair that answers the most queries while
precision stays at or above 98%. Test only reads the saved pair.

**5. What does a wrong match cost, compared with an abstain?**
A wrong match silently adds one company's exposure to another, and nobody looks again.
An abstain goes to a person, who spends a minute on it. So we favour precision.

**6. Why use an LLM only for low-margin cases?**
That is where string similarity has run out of signal and world knowledge may help.
It is a small share of queries, so cost and latency stay low. Its answer is checked in
code, and it can always say "none".

**7. How would this scale to millions of names?**
Swap the dense similarity step for an approximate nearest-neighbour index (for example
FAISS or a vector database) over the same n-gram vectors, keep the re-rank on the top 20,
and cache frequent queries. The index is built offline; the API only loads it.

**8. What would you add next?**
Use more fields than the name: state, industry (SIC), address, website. Add an acronym
rule. Learn the final score with a small model (logistic regression on the fuzzy scores,
TF-IDF score and field matches) from labelled pairs. Hand-label real GP-report names to
check that the synthetic test matches reality.

**9. (Bonus) What is point-in-time matching and why does it matter?**
Names move between companies. "Walt Disney Co" meant one CIK before 2019 and another after.
A 2016 report should be matched with the names valid in 2016, or the history gets rewritten.
