# Operating the data lake

Bronze (raw documents) → silver (Parquet tables) → gold (views). Built by `make lake`, checked by
`make lake-validate`, deployed by `make deploy-lake` (S3 + Glue + Athena).

## 1. Backtesting

- **Point in time:** each fund report has an as-of date, and names are resolved as of that date.
  A quarterly rebuild keeps each quarter's tables, so any question can be re-asked "as of" a
  past quarter, the same way the classifier was backtested by year.
- **Reproducibility (measured):** a clean rebuild (`make lake-clean && make lake`) gave the
  same 50 pass / 1 warn / 0 fail and the same row counts.
- **Not yet:** more than one quarter of fund reports. The next step is two or more quarters,
  with checks that holdings carry forward and that renames between quarters still resolve.

## 2. Precision or recall

**Chosen: precision.** The lake is the system of record, so a wrong link or number is worse than a
gap.
- Holdings that the matcher can't resolve are stored with `cik` NULL and `match_decision =
  abstain`. They are never forced to a company.
- Metrics keep XBRL and extracted values side by side. Extracted values that failed checks are
  stored with `status=needs_human_review`, not as facts.

**Measured:** matching precision and recall are both 1.000 on 54 holdings. Extracted vs. XBRL
agrees within 1% for 82% of revenue and 84% of net income values.

**Warning:** harder names ("Meta Platforms" → SKYX Platforms) still pass today's thresholds. For
money, tune on adversarial names (see `docs/DIALS.md`).

**To flip toward recall:** accept lower-scoring matches into a `match_candidates` table for review,
and load extracted values that failed checks with a flag.

## 3. Cost

- **Build:** about $0.46 per full rebuild (extraction on 50 filings, Haiku), then $0 with the cache.
  PDF parsing, matching and Parquet writing are local and free.
- **AWS (idle):**
  - S3: cents a month for MBs of Parquet.
  - Glue catalog: free at this size.
  - Athena: $0 until queried.
- **AWS (per query):** $5 per TB scanned with a 10 MB minimum. That's fractions of a cent per
  query here; the workgroup cuts off any query at 1 GB.

## 4. Monitors

**In place:** `make lake-validate` exits non-zero on any hard failure (schema, nulls, keys,
referential integrity, PDF totals vs. truth, NAV vs. fair value) and writes `results/validate.csv`.

**Recommended:**
- Run validate in CI on every rebuild, and block publishing to S3 on a hard failure.
- Track the abstain rate on holdings and the extracted-vs-XBRL agreement per quarter.
- Alert when a parsed fund's totals don't match its printed Total row (already a hard check).

## 5. Guardrails

**In place:**
- The parser **never reads the truth files**.
- It raises an error if parsed rows don't add up to the PDF's printed Total.
- Matching uses as-of dates.
- Schema contract checks run on DuckDB and Parquet.
- Synthetic documents are marked SYNTHETIC on every page.
- The S3 bucket blocks public access, is encrypted, accepts TLS only, and auto-empties on teardown.

**Recommended:**
- Object versioning and a lake-wide data contract test in CI.
- PII scanning on real fund documents before they land in bronze.
