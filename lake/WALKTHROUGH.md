# The data lake: fund reports + 10-Ks in one queryable place

This folder builds a small data lake from documents: five **synthetic** fund reports (PDF)
and the **real** 10-K filings and XBRL numbers of the ~50 public companies those funds hold.
It ends in a handful of tables (Parquet files plus one DuckDB database) that an analyst, or
the LangGraph agent in `sprint/ml/agent/`, can query with SQL.

Read this top to bottom once; each section says *what* the piece does and *why* it is built that way.

---

## 1. The problem

A limited partner (LP: a pension fund, an endowment) invests in many private funds. Every
quarter each fund's general partner (GP) sends a PDF report: the fund's value (NAV), how much
money it has called, and a schedule of what it owns.

Three things make these reports hard to use together:

1. **Names are inconsistent.** One GP writes "Facebook", another "Meta Platforms, Inc.",
   a third "FACEBOOK INC", and someone types "JPMORGAN CHAE & CO". The LP needs to know these
   are one company, or it can't add up its exposure.
2. **Names change over time.** Rexnord became Zurn Elkay Water Solutions in 2021. A 2019
   report correctly says "Rexnord Corp". Matching it to today's name needs the *name history*
   and the *report date*.
3. **The useful facts live in other documents.** "How much of my money sits in companies that
   lost money last year?" needs the holdings (from the GP PDF) joined to the companies'
   financials (from their 10-Ks or XBRL).

The lake solves this: read every document once, resolve every name to one ID (the SEC's CIK),
and store the result as tables that join on that ID.

## 2. Bronze, silver, gold (and why three layers)

```
 documents                bronze (as received)          silver (clean tables)       gold (answers)
 ---------                --------------------          ---------------------       --------------
 GP report PDFs   ──────► fund_reports/F01.pdf ──parse──► funds, holdings ──┐
 10-K text        ──────► filings/{cik}_{year}.json ───► doc_chunks         ├──► fund_exposure
                                     └──LLM extraction──► metrics ──────────┼──► company_financials
 SEC XBRL facts   ──────► xbrl/{cik}.json ─────────────► metrics            ├──► metric_reconciliation
 SEC name registry──────► reference/entities.jsonl ────► companies ─────────┘
                                                        (entity master)
```

- **Bronze** is a copy of each source document, untouched. If a later step has a bug, you fix
  the code and rebuild from bronze: no need to go back to the GP or the SEC. It is also the
  audit trail: every row in silver says which bronze file it came from (`source_doc`).
- **Silver** is the same information as clean, typed tables with one meaning per column
  (dollars, not "USD 000s"; a CIK, not a name string). This is where the data-quality rules live.
- **Gold** is shaped for questions: one view per common question (exposure by sector,
  financials by year, LLM-vs-XBRL agreement). Gold is cheap to change because it is just SQL
  over silver.

Why not one big step? Because each layer has a different reason to change. A new PDF layout
changes the parser (bronze → silver). A new analyst question changes a view (silver → gold).
Keeping them apart means a change in one place doesn't ripple everywhere.

## 3. The entity master, and how matching feeds it

The **entity master** is the `companies` table: one row per real company, keyed by CIK, with its
current name, sector, SIC code, and every name it has used (`aliases`, ' | ' joined). Every
other table points at it by CIK.

The **matching module** (`sprint/ml/matching/`, reused as is) turns a reported name into a CIK:

1. `normalize()` cleans the name: lowercase, "&" → "and", drop "Inc"/"Corp" at the end,
   unify "Hldgs"/"Holdings".
2. TF-IDF on character pieces (3 to 5 letters) finds the 20 most similar companies out of
   ~14,000 aliases (SEC names, former names, ~10k listed filers). Character pieces survive
   typos: "chae" and "chase" still share "cha".
3. rapidfuzz re-scores those 20 carefully (0–100).
4. `decide()` answers only if the best score ≥ 70 and it leads the runner-up (thresholds tuned
   on the matching dev set). Otherwise it **abstains**: `cik` is NULL and a person reviews it.
5. **`as_of`**: only names that were valid on the report date count. This is what lets
   "Rexnord Corp" in a 2019 report resolve to CIK 1439288 without confusing it with anything
   else, and what stops a 2019 report's names from matching a 2023 name.

In this lake the company list itself comes from the documents: `ingest.py` reads the PDFs,
resolves the names, and pulls 10-Ks and XBRL only for the CIKs it found. That is how it would
work for real: the LP only knows what the GP reports say.

## 4. Each file, part by part

| File | What it does |
|---|---|
| `common.py` | Paths, the **schema contract** (`SILVER_TABLES`), sector code ↔ label, `chunk_text()`, and thin wrappers that reuse the matching and extraction modules. |
| `funds.py` | Invents 5 funds (seed 14), picks real companies, writes one PDF per fund and the truth files. |
| `parse_pdf.py` | PDF → summary dict + holdings rows + commentary text. Never reads the truth. |
| `ingest.py` | Bronze: resolve the PDFs' names, copy each company's 10-K sections, XBRL facts and SEC registry entry. |
| `build.py` | Silver + gold: builds the five tables, runs the LLM extraction graph, writes Parquet and `lake.duckdb`. |
| `validate.py` | Data-quality checks against the truth files; exits 1 on a hard failure. |
| `golden_questions.py` | ~55 analyst questions with expected answers computed from the truth files and XBRL. |
| `lake.mk` | Make targets for every stage. |
| `../tests/test_lake.py` | Unit tests on tiny inputs: name variants, PDF round trip, schema, checks. |

### `funds.py`: the synthetic universe
- `FUNDS`: five invented funds ("Larkspur Hollow Growth Fund II", ...) and the CIKs each holds:
  54 holdings, 50 distinct companies, 4 held by two funds. Every company has a fiscal-2019 10-K
  in the corpus and XBRL revenue, net income and total assets for 2019.
- `former_name_on(company, as_of)`: was a *different* name valid on 2019-12-31? Ten held
  companies renamed themselves after that date (Facebook → Meta, Rexnord → Zurn Elkay,
  FleetCor → Corpay, Allegheny Technologies → ATI, Anthem → Elevance, Piper Jaffray →
  Piper Sandler, Gray Television → Gray Media, Realogy → Anywhere, NortonLifeLock → Gen Digital,
  Newmont Goldcorp → Newmont), so the 2019 report uses the old name. A few others renamed
  during 2019 (HCP → Healthpeak, Ensco Rowan → Valaris, Symantec → NortonLifeLock), which
  the commentary mentions.
- `reported_name()`: otherwise picks a form at random: the current name, a suffix swap
  ("Corp" → "Inc"), a case change, an abbreviation, or a one-letter typo. It **reuses**
  `make_variant()` from `matching/golden.py`, so the lake and the matcher's test set use the
  same idea of a realistic edit. The kind is recorded in the truth file.
- `make_fund()`: cost, fair value (0.55x–2.4x cost), ownership %, then fund totals: called
  capital a bit above total cost (fees), commitment above called, NAV = fair value + 1–4% cash.
  All amounts are whole thousands, because the PDF prints thousands.
- `commentary()`: 2–3 paragraphs built only from real data: XBRL revenue growth, net losses,
  and renames from the SEC name history.
- `write_fund_pdf()`: reportlab. Cover with a red SYNTHETIC banner, summary table + commentary
  on page 2, schedule of investments with a Total row on page 3. The banner is on every page.

### `parse_pdf.py`: documents → rows
- `pdfplumber` finds ruled tables by following their grid lines (`extract_tables()`).
- Tables are recognized by their **header**, not their position ("Item" → summary,
  "Portfolio company" → holdings), so a report that moves a table still parses.
- `scale_of()` reads the unit from the header: "(USD 000s)" → × 1,000.
- The printed **Total** row is compared with the sum of the rows. If they differ, `ParseError`:
  a half-read report must never reach the lake quietly.
- The commentary is the page text after "Manager Commentary", minus the banner lines.

### `ingest.py`: bronze
- `resolved_ciks()`: parse each PDF, match each name as of its report date.
- `copy_filings()`: streams the raw EDGAR-CORPUS files once (reusing `extraction/common.load_texts`)
  and writes sections 1, 1A, 7 (what analysts read) plus 8 and 15 (what the extraction graph
  reads) for fiscal 2019 and 2018. Only 16 companies have a 2018 10-K in this corpus; that is fine,
  since XBRL covers 2018 numbers.
- `copy_xbrl()`, `write_reference()`: XBRL facts and SEC registry rows for the same CIKs.

### `build.py`: silver and gold
- `companies_frame()`: the entity master, with `aliases` (so `aliases ILIKE '%rexnord%'` works).
- `fund_frames()`: parsed PDFs → `funds` and `holdings`; each reported name goes through the
  matcher (`resolve()` in `common.py`), giving `cik`, `matched_name`, `match_score`, `match_decision`.
- `xbrl_rows()`: five metrics × two years from XBRL, using the extraction module's
  `match_filing()` rule. A missing number is a row with `status = 'missing'`, not an absent row.
  If there's no 2018 XBRL revenue, it takes the prior-year column of the 2019 10-K.
- `run_extractions()`: runs the **existing LangGraph extraction flow** (`extraction/graph.py`:
  locate → extract → normalize → check → analyze, with one retry and a human-review route)
  on each fiscal-2019 10-K with Claude Haiku 4.5 on Bedrock. Each result is cached in
  `data/lake/work/extraction/`, so a rebuild costs nothing.
- `extracted_rows()` / `clean_value()`: the LLM's numbers are validated **in code** before
  they become rows: only finite floats survive. `status` is `ok` only if the whole graph
  run passed its checks.
- `chunk_rows()`: ~1,500-character chunks, no overlap, ids like `10k-1326801-2019-item1a-003`
  and `fund-F01-p2-001`. Same text → same ids, so the golden questions can cite them.
- `write_lake()`: deletes and recreates `lake.duckdb`, creates each table **from the contract**
  (`CREATE TABLE ... (cik BIGINT, ...)` + `CAST` on insert), exports each to
  `silver/{table}/data.parquet`, creates the gold views and exports those too.

### `validate.py`: does the lake tell the truth?
Hard checks (exit 1): schema = contract; row counts; no null keys; unique keys; every matched
CIK exists in `companies`; per fund, holdings count and cost and fair-value totals equal the
truth; fund summary numbers equal the truth; **matching precision ≥ 0.95**; NAV is between
100% and 110% of the holdings' fair value.
Soft checks (warn): matching recall ≥ 0.90; LLM-extracted numbers within 1% of XBRL for ≥ 80%
of the runs that passed their checks.

Precision and recall, in plain words: **precision** = of the names the matcher answered, how
many it got right (a wrong answer puts a company in the wrong fund, so it's a hard check).
**Recall** = of all holdings, how many it got right (an abstain lowers recall but is safe:
a person fixes it).

### `golden_questions.py`: an exam for the agent
Seven question types: entity resolution, fund look-through, aggregation, financials,
cross-cutting, text (risk factors), unanswerable. The expected answers are computed from the
**truth files, XBRL and the raw 10-K text**, never from the lake tables. If the lake were wrong,
an agent that faithfully reads the lake would fail, which is what a golden set should catch.
Text questions list the chunk-id prefix and the exact chunk ids that contain the keyword.

## 5. Tool choices, and the alternatives

**Parquet vs. CSV vs. a database.**
Parquet stores each column together, compressed, with its type. A query that needs 2 of 13
columns reads only those 2, and "BIGINT" stays a BIGINT (CSV would turn CIK 0000320193 into
320193 or a float, and dates into strings). A database (Postgres) would also work, but then
the data is locked inside one server. Parquet files are plain files that any engine reads:
DuckDB, Athena, Spark, pandas, BigQuery. CSV is still fine for the small truth files, where a
human opening them in a spreadsheet matters more than types.

**DuckDB locally vs. Athena in the cloud (same files).**
DuckDB is a database that runs inside the Python process: no server, reads Parquet directly,
fast on a laptop for millions of rows. Athena is AWS's serverless SQL over files in S3. The point
of the layout `silver/{table}/data.parquet` is that you could `aws s3 sync data/lake/ s3://…`,
point an AWS Glue table at each folder, and run the **same SQL** in Athena. You'd move when the
data no longer fits on one machine, or when many people need to query it at once with access
control. At 50 companies, DuckDB is the right tool: zero cost, zero setup.

**pdfplumber vs. Textract vs. an LLM for PDFs.**
- *pdfplumber* reads the text and lines that are already in a digital PDF. Free, exact,
  deterministic, but it needs a machine-made PDF (not a scan) and a known layout.
- *Amazon Textract* (or Google Document AI) does OCR plus table detection. Needed for
  scanned reports and varied layouts; costs about $0.015 per page for tables and still needs
  checking.
- *An LLM* (send the page, ask for JSON) handles any layout and even odd wording, but it can
  silently drop or invent a row, and costs per page.
Here the PDFs are digital and uniform, so pdfplumber wins. For real GP reports the usual answer
is layered: pdfplumber first, Textract for scans, an LLM only for the leftovers, and always the
Total-row check, whichever tool read the table.

**reportlab for the synthetic docs.**
It draws PDFs from Python with real ruled tables, so the parser faces a real PDF (not a text
file pretending to be one). `invariant=1` removes the timestamp, so the same seed gives the same
bytes. Alternatives: HTML → PDF (weasyprint) is nicer to style but adds system dependencies;
python-docx → PDF needs Word or LibreOffice.

**XBRL vs. extracted metrics.**
XBRL is the company's own machine-readable tagging of its statements: exact, cheap, and the
lake's trusted source (the gold `company_financials` uses only XBRL). It still has gaps (some
companies use custom tags; banks have no "Revenues"). The LLM extraction reads the 10-K text:
it can find numbers XBRL doesn't tag, and it's what you'd do for documents with no XBRL at all
(private-company financials from a GP). Keeping both, side by side with `source` and
`evidence`, and reconciling them in `metric_reconciliation` tells you how far you can trust the
LLM before you rely on it where there is no XBRL to check against.

## 6. Weaknesses (say these before an interviewer does)

- **Synthetic documents are cleaner than real GP reports.** One layout, digital text, ruled
  tables, no scans, no footnotes, no merged cells, no "see prior quarter". The parser would
  need real work for real reports.
- **The reported names are easy for this matcher.** Precision and recall are both 1.00 on the
  54 holdings. Harder names break it: with `as_of = 2019-12-31`, "Meta Platforms" (a name from
  the future) matches **SKYX Platforms** at 71.3, and "Symantec" (renamed two months earlier)
  matches **Stantec** at 80. Both pass the tuned threshold (score ≥ 70, margin ≥ 0). Those
  thresholds were tuned for coverage on a different set; a lake feeding money decisions would
  want a higher bar plus human review of close calls.
- **Small:** 5 funds, 54 holdings, 50 companies, two fiscal years.
- **The 10-K tables were stripped** in EDGAR-CORPUS, so the LLM only sees numbers that appear in
  sentences. Result: total assets were never found (it lives only in the balance-sheet table),
  revenue was found in 24 of 50 filings, net income in 19. Where the graph passed its checks,
  18 of 22 revenues and 16 of 19 net incomes are within 1% of XBRL.
- **Sectors come from one dataset** (`companies.csv`), whose taxonomy is a little odd
  (UnitedHealth and Anthem under "financials", Tapestry under "materials").
- **No history:** one quarter per fund. A real lake keeps every quarter and versions
  the entity master (who was matched to what, when).

## 7. Run it

From `sprint/ml/`. The main Makefile includes `lake/lake.mk`, so these are ordinary `make` targets.

```
make fund-docs        # 5 PDFs + truth files                      ~2 s
make lake-ingest      # bronze: 66 filings, 50 XBRL files          ~5 s
make lake-build       # silver + gold; LLM on 50 filings           a few minutes the first time, ~$0.46
make lake-validate    # checks; exit 1 on a hard fail              ~2 s
make lake-golden      # questions.jsonl                            ~3 s
make lake             # all five, in order
make lake-clean       # delete data/lake/ (also the cached LLM runs: the next build pays again)
uv run python -m pytest tests/test_lake.py -q    # 12 tests, no network
```
`lake-build` needs an AWS profile with Bedrock access: `make lake-build PROFILE=<your-profile>`
(or export `AWS_PROFILE`). Without one: `uv run python -m lake.build --no-llm` builds everything
except the `extracted` metric rows.

Expected output (abridged):
```
$ make fund-docs
5 SYNTHETIC fund reports -> data/lake/bronze/fund_reports
54 holdings, 50 distinct companies, 4 held by two funds
reported-name kinds: current 23, suffix 10, former_name 9, typo 9, case_punct 2, former_variant 1

$ make lake-ingest
50 companies resolved from 5 fund reports
bronze: 66 filings, 50 XBRL files, 50 reference entities

$ make lake-build
extraction: 0 cached, 50 to run on us.anthropic.claude-haiku-4-5-20251001-v1:0
  1/50  cik 1018963: done  ($0.0060)   ...
  companies 50 | funds 5 | holdings 54 | metrics 750 | doc_chunks 8490
  company_financials 100 | fund_exposure 15 | metric_reconciliation 250
extraction spend (all cached runs, list price): $0.464

$ make lake-validate
matching on 54 holdings: precision 1.000, recall 1.000
WARN  extracted vs xbrl:total_assets: 0.0 (want 0.8) 0/0 status=ok rows within 1%; 50 missing
50 pass, 1 warn, 0 hard fail -> results/validate.csv

$ make lake-golden
57 golden questions: entity_resolution 10, look_through 8, aggregation 11, financials 10,
cross_cutting 6, text 7, unanswerable 5
```

Try a query:
```
uv run python -c "import duckdb; print(duckdb.connect('data/lake/lake.duckdb', read_only=True).sql(
  \"SELECT fund_id, reported_name, matched_name, match_score FROM holdings WHERE match_score < 100\"))"
```

## 8. Interview questions

**1. Why not just load the PDFs' tables straight into one table and query that?**
Because the names don't join. "Facebook" in one report and "META PLATFORMS INC" in another are
different strings; the sum of "my exposure to Meta" would be wrong. The entity master gives
every row a CIK, and everything joins on that.

**2. How do you know the PDF parser didn't miss a row?**
Two ways. Inside the document: the parser compares the rows' sum to the report's printed Total
and refuses to continue if they differ. Against the truth: `validate.py` checks the count and
totals per fund. In production you'd only have the first one, which is why it lives in the
parser, not in the test.

**3. What does `as_of` do, and why does it matter?**
It hides names that weren't valid on the report date. A 2019 report saying "Rexnord Corp" should
resolve to the company then called Rexnord. It also stops "the future": a 2019 report can't
legitimately say "Meta Platforms". The cost: a stale or future name gets a wrong match instead
of an exact one (SKYX for "Meta Platforms"), which is why abstaining and human review matter.

**4. Which matching error is worse, and how did you set the checks?**
A wrong match. It silently moves money into the wrong company and every aggregate inherits the
error. An abstain is visible (`cik IS NULL`) and a person fixes it. So precision is a hard
check (≥ 0.95) and recall is a soft one.

**5. Why keep both XBRL and LLM-extracted numbers?**
XBRL is the trusted source for public companies; the LLM is what you'd use when there is no
XBRL (a private company's financials in a GP report). Running both where XBRL exists measures
the LLM: 18 of 22 passed-check revenues within 1%. That number tells you how much human review
to plan for on documents with no answer key.

**6. How would you move this to AWS?**
Same files, different engine. Bronze, silver and gold go to S3 under the same paths; a Glue
crawler or `CREATE EXTERNAL TABLE` registers each Parquet folder; Athena runs the same SQL views.
The build becomes a Step Functions or Glue job triggered when a PDF lands in S3; Bedrock is
already the LLM. Add partitions (`fund_id`, `as_of`) once there are many quarters, and Lake
Formation for who may see which fund.

**7. What would you add for real GP reports?**
Textract (or an LLM) as a fallback for scans and new layouts; a per-GP parser config; a review
queue for abstains and failed Total checks; history (every quarter, and a log of who was matched
to what, when); and data contracts with tests in CI so a schema change breaks loudly.

**8. How do you test an agent that answers from this lake?**
With the golden questions: answers computed from the truth files and XBRL, never from the lake,
grouped by type so you see *where* it fails (entity resolution vs. arithmetic vs. text).
Numbers have tolerances; sets are compared as sets; text answers are graded on cited chunk ids
and keywords; "unanswerable" questions check it says "I don't know" instead of inventing one.
