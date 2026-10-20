"""Silver and gold: turn the bronze documents into tables, as Parquet and in DuckDB.

Run:        AWS_PROFILE=<your-profile> uv run python -m lake.build
No LLM:     uv run python -m lake.build --no-llm      (skips the `extracted` metric rows)

Silver tables (one Parquet file each, data/lake/silver/{table}/data.parquet)
  companies   the entity master: one row per SEC company the reports resolved to
  funds       one row per fund report (parsed from the PDF summary table)
  holdings    one row per reported holding, with the matched CIK (or an abstain)
  metrics     five financial metrics per company-year, from XBRL and from the
              LLM extraction graph, side by side, each with its evidence
  doc_chunks  ~1,500-character pieces of the 10-K text and the fund commentary

Gold views (data/lake/gold/{view}/data.parquet, and views in DuckDB)
  company_financials, fund_exposure, metric_reconciliation

Everything is rebuilt from bronze each run (delete and recreate), so a rerun gives
the same tables. The only thing kept between runs is the cache of LLM extraction
results in data/lake/work/extraction/, so a rebuild does not pay for the LLM twice.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

from lake.common import (
    BRONZE_XBRL,
    CHUNKED_SECTIONS,
    DUCKDB_FILE,
    FILINGS,
    FISCAL_YEARS,
    FUND_REPORTS,
    GOLD,
    GOLD_VIEWS,
    LAKE,
    METRICS,
    REFERENCE,
    SILVER,
    SILVER_TABLES,
    WORK,
    chunk_text,
    load_matcher,
    resolve,
    use_extraction_modules,
    xbrl_values,
)
from lake.parse_pdf import parse_fund_pdf

EXTRACT_YEAR = 2019  # the LLM reads each company's fiscal-2019 10-K
XBRL_TAGS = {"net_income": "NetIncomeLoss", "total_assets": "Assets",
             "operating_income": "OperatingIncomeLoss",
             "cash": "CashAndCashEquivalentsAtCarryingValue"}


# ---- companies: the entity master ---------------------------------------------

def companies_frame(entities: list[dict]) -> pd.DataFrame:
    """One row per company. `aliases` = current name + every former name, ' | ' joined,
    so a plain SQL `aliases ILIKE '%facebook%'` finds Meta."""
    rows = []
    for e in entities:
        names = [e["name"]] + [f["name"] for f in e["former_names"]]
        aliases = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))  # dedupe
        rows.append({"cik": e["cik"], "name": e["name"], "sector": e.get("sector"),
                     "sic": e.get("sic"), "aliases": " | ".join(aliases)})
    return pd.DataFrame(rows)


# ---- funds and holdings: from the PDFs, names resolved by the matcher ----------

def fund_frames(parsed: list[dict], matcher) -> tuple[pd.DataFrame, pd.DataFrame]:
    funds, holdings = [], []
    for p in parsed:
        f = p["fund"]
        funds.append({**{k: f[k] for k in ("fund_id", "fund_name", "vintage", "commitment_usd",
                                           "called_usd", "nav_usd", "as_of")},
                      "source_doc": p["source_doc"]})
        names = [h["reported_name"] for h in p["holdings"]]
        for h, m in zip(p["holdings"], resolve(matcher, names, f["as_of"])):
            holdings.append({"fund_id": f["fund_id"], **h, **m, "as_of": f["as_of"],
                             "source_doc": p["source_doc"]})
    return pd.DataFrame(funds), pd.DataFrame(holdings)


# ---- metrics from XBRL ----------------------------------------------------------

def xbrl_rows(cik: int, company: dict) -> list[dict]:
    """Five metrics x the fiscal years we keep. A missing number is a row with
    status 'missing', not an absent row, so "we looked and it isn't there" is visible."""
    rows = []
    by_year = {y: xbrl_values(company, y) for y in FISCAL_YEARS}
    for year in FISCAL_YEARS:
        v = dict(by_year[year])
        evidence = {m: f"us-gaap:{XBRL_TAGS[m]} in filing {v.get('accn')}" for m in XBRL_TAGS}
        evidence["revenue"] = f"us-gaap:{v.get('revenue_tag')} in filing {v.get('accn')}"
        # No 10-K tagged for this year (common for 2018 here): the next year's 10-K
        # still reports last year's revenue as its comparative column.
        nxt = by_year.get(year + 1) or {}
        if v.get("revenue") is None and nxt.get("revenue_prior") is not None:
            v["revenue"] = nxt["revenue_prior"]
            evidence["revenue"] = (f"us-gaap:{nxt.get('revenue_tag')} prior-year column of "
                                   f"filing {nxt.get('accn')}")
        for m in METRICS:
            value = v.get(m)
            rows.append({"cik": cik, "fiscal_year": year, "metric": m,
                         "value": None if value is None else float(value), "source": "xbrl",
                         "evidence": evidence[m] if value is not None else None,
                         "status": "ok" if value is not None else "missing",
                         "source_doc": f"bronze/xbrl/{cik}.json"})
    return rows


# ---- metrics from the LLM extraction graph -------------------------------------

def run_extractions(filings: dict, entities: dict, xbrl: dict, workers: int) -> dict:
    """Run extraction/graph.py on each fiscal-2019 10-K; cache each result as JSON.

    Returns {cik: result}. Cached results are reused, so only new filings cost money.
    """
    use_extraction_modules()
    import llm  # extraction/llm.py: Bedrock, Claude Haiku 4.5, retries only 429/5xx
    from analysis import sector_reference
    from graph import build_graph

    cache_dir = WORK / "extraction"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    for (cik, year), doc in filings.items():
        path = cache_dir / f"{cik}_{year}.json"
        if year != EXTRACT_YEAR:
            continue
        if path.exists():
            results[cik] = json.loads(path.read_text())
        else:
            todo.append((cik, doc))
    print(f"extraction: {len(results)} cached, {len(todo)} to run on {llm.MODEL_ID}")
    if not todo:
        return results
    graph, ref = build_graph(), sector_reference()

    def one(cik: int, doc: dict) -> dict:
        e = entities[cik]
        prior = xbrl_values(xbrl[cik], EXTRACT_YEAR).get("revenue_prior")
        t0 = time.perf_counter()
        try:
            s = graph.invoke({"cik": cik, "year": EXTRACT_YEAR, "name": e["name"],
                              "sector": e.get("sector") or "", "sections": doc["sections"],
                              "prior_revenue": prior,
                              "sector_ref": ref.get(e.get("sector"), {})})
        except Exception as err:  # noqa: BLE001 (e.g. circuit breaker open): record, go on
            s = {"status": "needs_human_review", "reason": f"{type(err).__name__}: {err}"[:300]}
        return {"cik": cik, "year": EXTRACT_YEAR, "status": s.get("status"),
                "reason": s.get("reason"), "values": s.get("values") or {},
                "extracted": s.get("extracted") or {}, "trace": s.get("trace", []),
                "tokens_in": s.get("tokens_in", 0), "tokens_out": s.get("tokens_out", 0),
                "seconds": round(time.perf_counter() - t0, 1)}

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, cik, doc): cik for cik, doc in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            (cache_dir / f"{r['cik']}_{EXTRACT_YEAR}.json").write_text(json.dumps(r, indent=1))
            results[r["cik"]] = r
            print(f"  {i}/{len(todo)}  cik {r['cik']}: {r['status']}  "
                  f"(${llm.cost(r['tokens_in'], r['tokens_out']):.4f})")
    return results


def clean_value(v) -> float | None:
    """Validate an LLM-derived number in code: a finite float, or nothing."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def extracted_rows(run: dict) -> list[dict]:
    """One row per metric. `ok` only when the whole graph run passed its checks."""
    rows = []
    for m in METRICS:
        value = clean_value(run["values"].get(m))
        evidence = run["extracted"].get(f"{m}_evidence")
        evidence = evidence[:300] if isinstance(evidence, str) and value is not None else None
        status = "missing" if value is None else (
            "ok" if run["status"] == "done" else "needs_human_review")
        rows.append({"cik": run["cik"], "fiscal_year": run["year"], "metric": m, "value": value,
                     "source": "extracted", "evidence": evidence, "status": status,
                     "source_doc": f"bronze/filings/{run['cik']}_{run['year']}.json"})
    return rows


# ---- doc_chunks ----------------------------------------------------------------

def chunk_rows(filings: dict, parsed: list[dict]) -> list[dict]:
    """Stable ids: 10k-{cik}-{year}-item1a-003, fund-{fund_id}-p2-001."""
    rows = []
    for (cik, year), doc in sorted(filings.items()):
        for section, item in CHUNKED_SECTIONS.items():
            for i, text in enumerate(chunk_text(doc["sections"].get(section) or ""), 1):
                rows.append({"chunk_id": f"10k-{cik}-{year}-{item}-{i:03d}",
                             "doc_id": f"10k-{cik}-{year}", "doc_type": "10-K", "cik": cik,
                             "fund_id": None, "fiscal_year": year, "section": item, "text": text})
    for p in parsed:
        fid = p["fund"]["fund_id"]
        for c in p["commentary"]:
            for i, text in enumerate(chunk_text(c["text"]), 1):
                rows.append({"chunk_id": f"fund-{fid}-p{c['page']}-{i:03d}",
                             "doc_id": f"fund-{fid}", "doc_type": "fund_report", "cik": None,
                             "fund_id": fid, "fiscal_year": None, "section": "commentary",
                             "text": text})
    return rows


# ---- write: DuckDB tables + Parquet files --------------------------------------

GOLD_SQL = {
    # Wide financials per company-year, XBRL only (the trusted source), with growth.
    "company_financials": """
        WITH x AS (
            SELECT cik, fiscal_year,
                   max(value) FILTER (WHERE metric = 'revenue')      AS revenue,
                   max(value) FILTER (WHERE metric = 'net_income')   AS net_income,
                   max(value) FILTER (WHERE metric = 'total_assets') AS total_assets
            FROM metrics WHERE source = 'xbrl' GROUP BY cik, fiscal_year)
        SELECT x.cik, c.name, c.sector, x.fiscal_year, x.revenue, x.net_income, x.total_assets,
               round(100 * (x.revenue / p.revenue - 1), 2) AS revenue_growth_pct
        FROM x JOIN companies c USING (cik)
        LEFT JOIN x AS p ON p.cik = x.cik AND p.fiscal_year = x.fiscal_year - 1""",
    # Look-through: what each fund holds, by sector, and the share of its NAV.
    "fund_exposure": """
        SELECT h.fund_id, f.fund_name, h.sector, count(*) AS n_holdings,
               sum(h.cost_usd) AS cost_usd, sum(h.fair_value_usd) AS fair_value_usd,
               round(100 * sum(h.fair_value_usd) / f.nav_usd, 2) AS pct_of_nav
        FROM holdings h JOIN funds f USING (fund_id)
        GROUP BY h.fund_id, f.fund_name, h.sector, f.nav_usd""",
    # The LLM's numbers next to XBRL's, for every company-year-metric it attempted.
    "metric_reconciliation": """
        SELECT x.cik, x.fiscal_year, x.metric, x.value AS xbrl_value, e.value AS extracted_value,
               round(100 * (e.value - x.value) / abs(x.value), 3) AS pct_diff,
               abs(e.value - x.value) <= 0.01 * abs(x.value) AS agrees_within_1pct
        FROM metrics x JOIN metrics e USING (cik, fiscal_year, metric)
        WHERE x.source = 'xbrl' AND e.source = 'extracted'""",
}
GOLD_ORDER = {"company_financials": "cik, fiscal_year", "fund_exposure": "fund_id, sector",
              "metric_reconciliation": "cik, fiscal_year, metric"}


def write_lake(frames: dict[str, pd.DataFrame], lake_dir: Path = LAKE) -> None:
    """Create every table with the exact contract types, then export Parquet.

    The DuckDB file is deleted and rebuilt: tables are derived data, bronze is the source.
    """
    db = lake_dir / "lake.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    for table, cols in SILVER_TABLES.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols)})")
        df = frames[table].reindex(columns=[c for c, _ in cols])
        df = df.astype(object).where(df.notna(), None)  # NaN -> NULL
        con.register("incoming", df)
        casts = ", ".join(f"CAST({c} AS {t})" for c, t in cols)
        con.execute(f"INSERT INTO {table} SELECT {casts} FROM incoming")
        con.unregister("incoming")
        out = lake_dir / "silver" / table / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY (SELECT * FROM {table}) TO '{out}' (FORMAT parquet)")
    for view in GOLD_VIEWS:
        con.execute(f"CREATE VIEW {view} AS {GOLD_SQL[view]}")
        out = lake_dir / "gold" / view / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY (SELECT * FROM {view} ORDER BY {GOLD_ORDER[view]}) TO '{out}' "
                    "(FORMAT parquet)")
    con.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip the extraction graph")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    entities = [json.loads(line) for line in (REFERENCE / "entities.jsonl").read_text().splitlines()]
    by_cik = {e["cik"]: e for e in entities}
    parsed = [parse_fund_pdf(p) for p in sorted(FUND_REPORTS.glob("*.pdf"))]
    filings = {}
    for path in sorted(FILINGS.glob("*.json")):
        doc = json.loads(path.read_text())
        filings[(doc["cik"], doc["fiscal_year"])] = doc
    xbrl = {int(p.stem): json.loads(p.read_text()) for p in BRONZE_XBRL.glob("*.json")}

    funds, holdings = fund_frames(parsed, load_matcher())
    metrics = [r for cik in sorted(xbrl) for r in xbrl_rows(cik, xbrl[cik])]
    spend = 0.0
    if not args.no_llm:
        use_extraction_modules()
        import llm

        runs = run_extractions(filings, by_cik, xbrl, args.workers)
        metrics += [r for cik in sorted(runs) for r in extracted_rows(runs[cik])]
        spend = sum(llm.cost(r["tokens_in"], r["tokens_out"]) for r in runs.values())
    frames = {"companies": companies_frame(entities), "funds": funds, "holdings": holdings,
              "metrics": pd.DataFrame(metrics), "doc_chunks": pd.DataFrame(chunk_rows(filings, parsed))}
    write_lake(frames)

    con = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    print(f"\nwrote {DUCKDB_FILE.relative_to(LAKE.parent.parent)} and Parquet under "
          f"{SILVER.relative_to(LAKE.parent.parent)}, {GOLD.relative_to(LAKE.parent.parent)}")
    for t in list(SILVER_TABLES) + GOLD_VIEWS:
        print(f"  {t:22s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>6} rows")
    if not args.no_llm:
        print(f"extraction spend (all cached runs, list price): ${spend:.3f}")
    con.close()


if __name__ == "__main__":
    main()
