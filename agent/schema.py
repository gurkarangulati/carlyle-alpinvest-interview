"""The lake's schema contract, written once and used in three places:

  - describe_tables() shows it to the model, so it writes correct SQL;
  - the CDK LakeStack turns it into Glue tables (Athena reads Parquet through them);
  - the tests build a tiny DuckDB fixture with exactly these columns.

Types are written in Glue/Athena (Hive) names. DUCKDB_TYPE maps them for the fixture.
Every table is listed with its layer (silver = cleaned facts, gold = analysis-ready).
"""

# table -> (layer, one-line purpose, [(column, glue type, one-line description)])
TABLES: dict[str, tuple[str, str, list[tuple[str, str, str]]]] = {
    "companies": ("silver", "One row per public company (SEC filer).", [
        ("cik", "bigint", "SEC Central Index Key: the company id used everywhere"),
        ("name", "string", "current legal name as SEC writes it"),
        ("sector", "string", "sector label"),
        ("sic", "int", "SEC Standard Industrial Classification code"),
        ("aliases", "string", "former names and other spellings, joined by ' | '"),
    ]),
    "funds": ("silver", "One row per private-equity fund report.", [
        ("fund_id", "string", "fund id, e.g. used to join holdings"),
        ("fund_name", "string", "fund name"),
        ("vintage", "int", "vintage year"),
        ("commitment_usd", "double", "total LP commitments, USD"),
        ("called_usd", "double", "capital called so far, USD"),
        ("nav_usd", "double", "net asset value at as_of, USD"),
        ("as_of", "date", "report date"),
        ("source_doc", "string", "document the row came from"),
    ]),
    "holdings": ("silver", "One row per portfolio holding in a fund report.", [
        ("fund_id", "string", "fund that holds it (join to funds)"),
        ("reported_name", "string", "name as written in the fund report"),
        ("cik", "bigint", "matched company (NULL when the matcher abstained)"),
        ("matched_name", "string", "the matched company's name"),
        ("match_score", "double", "matcher score 0-100"),
        ("match_decision", "string", "'match' or 'abstain'"),
        ("sector", "string", "sector of the holding"),
        ("cost_usd", "double", "invested cost, USD"),
        ("fair_value_usd", "double", "fair value at as_of, USD"),
        ("ownership_pct", "double", "fund's ownership percentage"),
        ("as_of", "date", "report date"),
        ("source_doc", "string", "fund report it came from"),
        ("source_page", "int", "page in that report"),
    ]),
    "metrics": ("silver", "Long table of company financial metrics by fiscal year.", [
        ("cik", "bigint", "company"),
        ("fiscal_year", "int", "fiscal year"),
        ("metric", "string", "metric name, e.g. revenue, net_income, total_assets"),
        ("value", "double", "value in USD"),
        ("source", "string", "'xbrl' (official SEC data) or 'extracted' (read from 10-K text)"),
        ("evidence", "string", "sentence the extracted value came from"),
        ("status", "string", "extraction check status"),
        ("source_doc", "string", "filing it came from"),
    ]),
    "doc_chunks": ("silver", "Text passages of 10-Ks and fund reports (for search_docs).", [
        ("chunk_id", "string", "passage id; cite it"),
        ("doc_id", "string", "document id"),
        ("doc_type", "string", "'10-K' or 'fund_report'"),
        ("cik", "bigint", "company, for 10-K passages"),
        ("fund_id", "string", "fund, for fund-report passages"),
        ("fiscal_year", "int", "fiscal year, for 10-K passages"),
        ("section", "string", "section of the document"),
        ("text", "string", "the passage text"),
    ]),
    "company_financials": ("gold", "One row per company and fiscal year, wide.", [
        ("cik", "bigint", "company"),
        ("name", "string", "company name"),
        ("sector", "string", "sector"),
        ("fiscal_year", "int", "fiscal year"),
        ("revenue", "double", "revenue, USD"),
        ("net_income", "double", "net income, USD"),
        ("total_assets", "double", "total assets, USD"),
        ("revenue_growth_pct", "double", "revenue growth vs. prior year, percent"),
    ]),
    "fund_exposure": ("gold", "Each fund's exposure by sector.", [
        ("fund_id", "string", "fund"),
        ("fund_name", "string", "fund name"),
        ("sector", "string", "sector"),
        ("n_holdings", "bigint", "number of holdings in that sector"),
        ("cost_usd", "double", "total cost, USD"),
        ("fair_value_usd", "double", "total fair value, USD"),
        ("pct_of_nav", "double", "fair value as a percent of fund NAV"),
    ]),
    "metric_reconciliation": ("gold", "XBRL vs. text-extracted metric values, side by side.", [
        ("cik", "bigint", "company"),
        ("fiscal_year", "int", "fiscal year"),
        ("metric", "string", "metric name"),
        ("xbrl_value", "double", "official XBRL value, USD"),
        ("extracted_value", "double", "value extracted from the 10-K text, USD"),
        ("pct_diff", "double", "percent difference"),
        ("agrees_within_1pct", "boolean", "true when the two agree within 1%"),
    ]),
}

ALLOWED_TABLES = frozenset(TABLES)

DUCKDB_TYPE = {"bigint": "BIGINT", "int": "INTEGER", "double": "DOUBLE", "string": "VARCHAR",
               "date": "DATE", "boolean": "BOOLEAN"}


def describe_tables() -> str:
    """The schema as compact text for the model's prompt and the describe_tables tool."""
    lines = []
    for table, (layer, purpose, cols) in TABLES.items():
        lines.append(f"{table}  ({layer}) {purpose}")
        lines += [f"  - {c} {t}: {d}" for c, t, d in cols]
    return "\n".join(lines)
