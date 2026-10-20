"""Shared pieces of the data lake: paths, the schema contract, and small helpers.

Every lake script imports from here, so they all agree on where files live,
what each table looks like, and how text is cut into chunks.

Run every lake script from sprint/ml as a module, e.g.  uv run python -m lake.build
"""

import os
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]          # the repo root
DATA = ML / "data"
RAW = DATA / "raw" / "edgar"                      # EDGAR-CORPUS 10-Ks (read only)
XBRL_CACHE = DATA / "xbrl"                        # SEC XBRL facts, one JSON per company

LAKE = DATA / "lake"
BRONZE = LAKE / "bronze"                          # documents exactly as received
FUND_REPORTS = BRONZE / "fund_reports"            # {fund_id}.pdf
FILINGS = BRONZE / "filings"                      # {cik}_{year}.json
BRONZE_XBRL = BRONZE / "xbrl"                     # {cik}.json
REFERENCE = BRONZE / "reference"                  # SEC registry rows for the held companies
SILVER = LAKE / "silver"                          # clean tables, one folder per table
GOLD = LAKE / "gold"                              # analyst views, one folder per view
WORK = LAKE / "work"                              # caches (extraction runs), not a layer
GOLDEN = LAKE / "golden"                          # truth files + golden questions
DUCKDB_FILE = LAKE / "lake.duckdb"
RESULTS = Path(__file__).resolve().parent / "results"

AS_OF = "2019-12-31"          # the date every fund report is "as of"
FISCAL_YEARS = [2018, 2019]   # 10-K years we keep per company
METRICS = ["revenue", "net_income", "total_assets", "operating_income", "cash"]

# 10-K sections copied into bronze. 1, 1A and 7 are what analysts read (and what we
# chunk for search); 8 and 15 are copied too because the extraction graph reads them.
SECTIONS = ["section_1", "section_1A", "section_7", "section_8", "section_15"]
CHUNKED_SECTIONS = {"section_1": "item1", "section_1A": "item1a", "section_7": "item7"}
CHUNK_CHARS = 1_500

# ---- the schema contract -------------------------------------------------------
# Another program (the LangGraph agent in sprint/ml/agent/) queries these tables,
# so names and types are fixed here and build.py creates the tables from this dict.
SILVER_TABLES = {
    "companies": [("cik", "BIGINT"), ("name", "VARCHAR"), ("sector", "VARCHAR"),
                  ("sic", "INTEGER"), ("aliases", "VARCHAR")],
    "funds": [("fund_id", "VARCHAR"), ("fund_name", "VARCHAR"), ("vintage", "INTEGER"),
              ("commitment_usd", "DOUBLE"), ("called_usd", "DOUBLE"), ("nav_usd", "DOUBLE"),
              ("as_of", "DATE"), ("source_doc", "VARCHAR")],
    "holdings": [("fund_id", "VARCHAR"), ("reported_name", "VARCHAR"), ("cik", "BIGINT"),
                 ("matched_name", "VARCHAR"), ("match_score", "DOUBLE"),
                 ("match_decision", "VARCHAR"), ("sector", "VARCHAR"), ("cost_usd", "DOUBLE"),
                 ("fair_value_usd", "DOUBLE"), ("ownership_pct", "DOUBLE"), ("as_of", "DATE"),
                 ("source_doc", "VARCHAR"), ("source_page", "INTEGER")],
    "metrics": [("cik", "BIGINT"), ("fiscal_year", "INTEGER"), ("metric", "VARCHAR"),
                ("value", "DOUBLE"), ("source", "VARCHAR"), ("evidence", "VARCHAR"),
                ("status", "VARCHAR"), ("source_doc", "VARCHAR")],
    "doc_chunks": [("chunk_id", "VARCHAR"), ("doc_id", "VARCHAR"), ("doc_type", "VARCHAR"),
                   ("cik", "BIGINT"), ("fund_id", "VARCHAR"), ("fiscal_year", "INTEGER"),
                   ("section", "VARCHAR"), ("text", "VARCHAR")],
}
GOLD_VIEWS = ["company_financials", "fund_exposure", "metric_reconciliation"]
# The columns that must never be null, per table (validate.py checks them).
KEYS = {"companies": ["cik", "name"], "funds": ["fund_id", "as_of"],
        "holdings": ["fund_id", "reported_name", "match_decision"],
        "metrics": ["cik", "fiscal_year", "metric", "source", "status"],
        "doc_chunks": ["chunk_id", "doc_id", "doc_type", "text"]}


# ---- sectors -------------------------------------------------------------------
# The lake stores sector codes ("communications_media"); a fund report prints a
# label ("Communications Media"). These two functions convert both ways.

def sector_label(code: str) -> str:
    return code.replace("_", " ").title()


def sector_code(label: str) -> str:
    return "_".join(label.lower().split())


# ---- text chunks ---------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Cut text into pieces of about `size` characters, breaking only between words.

    No overlap, so chunk N is always the same text for the same document: the
    chunk ids stay stable across rebuilds, and a golden question can cite them.
    """
    chunks, current, length = [], [], 0
    for word in text.split():
        if current and length + 1 + len(word) > size:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + (1 if length else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---- reuse the extraction and matching modules ---------------------------------

def use_extraction_modules() -> None:
    """Make `import graph`, `import llm`, `from fetch_xbrl import ...` work.

    The extraction scripts import each other by plain name (they are run as
    `python extraction/x.py`), so their folder has to be on the import path.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # quiet a startup banner
    path = str(ML / "extraction")
    if path not in sys.path:
        sys.path.insert(0, path)


def xbrl_values(company: dict, year: int) -> dict:
    """The five metrics for one fiscal year from a company's cached XBRL facts.

    Reuses the extraction module's matching rule (pick the 10-K whose facts say
    fiscal year == `year`, annual periods only). Returns {} if nothing matches.
    Includes "revenue_prior" (the comparative column) and "accn" (the filing id).
    """
    use_extraction_modules()
    from fetch_xbrl import match_filing

    m = match_filing(company, year)
    return m if m.get("match_rule") == "form10K_fy_FY" else {}


def load_matcher():
    """The hybrid name matcher (TF-IDF shortlist, rapidfuzz re-score, abstain rule),
    with the thresholds tuned on the matching dev set."""
    from matching.matcher import load_matcher as _load

    return _load()


def resolve(matcher, names: list[str], as_of: str) -> list[dict]:
    """Reported names -> {cik, matched_name, match_score, match_decision}, one per name.

    `as_of` hides names that were not valid on the report date, so "Facebook Inc"
    resolves in a 2019 report even though the company is now "Meta Platforms".
    """
    from matching.matcher import decide

    out = []
    for name, cands in zip(names, matcher.search_many(names, "hybrid", k=5, as_of=as_of)):
        decision = decide(cands, matcher.min_score, matcher.min_margin)
        top = cands[0] if cands else None
        out.append({"cik": top["cik"] if decision == "match" else None,
                    "matched_name": top["name"] if decision == "match" else None,
                    "match_score": float(top["score"]) if top else 0.0,
                    "match_decision": decision})
    return out
