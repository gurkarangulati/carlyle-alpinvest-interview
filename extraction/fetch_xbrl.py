"""Ground-truth fixture: fetch SEC XBRL "company facts" and match them to our 10-Ks.

Smoke:  SEC_USER_AGENT="Your Name you@example.com" uv run python extraction/fetch_xbrl.py --limit 20
Run:    SEC_USER_AGENT="Your Name you@example.com" uv run python extraction/fetch_xbrl.py

The SEC asks every client to identify itself with a User-Agent ("name email").
We read it from the env var SEC_USER_AGENT and never store it in code.

Why XBRL is the answer key: since about 2011 every 10-K also files its statement
numbers as tagged data (XBRL). The SEC serves all of a company's tagged facts in one
JSON file. We keep only the five concepts we score, in data/xbrl/{cik}.json.

How a filing is matched to its facts (the matching rule)
--------------------------------------------------------
Best case: match by accession number (the SEC's ID for one filing, e.g.
0001493152-18-009029). Every XBRL fact carries the accession (`accn`) of the filing
it came from. Our EDGAR-CORPUS rows do NOT carry one: the `filename` field is just
"{cik}_{year}.htm". So we fall back to:

  1. Keep facts from form 10-K (or 10-K/A, 10-KT) with fiscal period "FY".
  2. Keep facts whose fiscal year `fy` equals the dataset's `year`. In this corpus
     `year` behaves like the fiscal year the 10-K reports, not the year it was
     filed (checked: the latest year discussed in Item 7 equals `year` in ~85% of
     filings, and the matched period ends in `year` for most matches; the script
     prints this check).
  3. Pick one filing among them: the original 10-K first, then the earliest filed.
     From here on we use only that filing's accession, so all numbers come from
     the same document.
  4. Within that filing, take the value for the latest period end. For flow items
     (revenue, net income) the period must be about one year long (330-400 days),
     which skips quarterly or transition-period numbers.
  5. Revenue has fallbacks: the first of the revenue tags below that has a value.
  6. The prior-year revenue (for growth) is the comparative column of the same
     filing: same accession, period ending about one year earlier.

Known failure cases:
  - No XBRL at all (404, or the company never tagged): no truth, filing is dropped.
  - The filer mis-tagged `fy` (e.g. tagged the filing year): no match, or the
    wrong year. The period_end column lets you spot this.
  - Company-specific tags: banks often have no "Revenues" (they report interest
    income), some firms use their own extension tags. Revenue is then missing.
  - Restatements: a later 10-K may restate a number. We use the original filing's
    values, which is what the text we read says, so this mostly helps us.
  - 52/53-week years and fiscal-year-end changes: a 10-KT can be shorter than a
    year and is skipped by the duration rule.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import httpx
import pandas as pd
from common import XBRL_DIR, load_companies

URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenuesNetOfInterestExpense"]
OTHER_TAGS = {"net_income": "NetIncomeLoss", "total_assets": "Assets",
              "operating_income": "OperatingIncomeLoss",
              "cash": "CashAndCashEquivalentsAtCarryingValue"}
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
FLOW_FIELDS = {"revenue", "net_income", "operating_income"}  # measured over a period
KEEP = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")


# ---------- fetching ----------

class RateLimiter:
    """Allow at most `per_second` request starts per second, across threads."""

    def __init__(self, per_second: float):
        self.gap = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            sleep_for = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.gap
        time.sleep(sleep_for)


def slim(facts_json: dict) -> dict:
    """Keep only our concepts, USD values, and annual-report forms. ~1 MB -> ~10 KB."""
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    keep = {}
    for tag in REVENUE_TAGS + list(OTHER_TAGS.values()):
        rows = gaap.get(tag, {}).get("units", {}).get("USD", [])
        rows = [{k: r.get(k) for k in KEEP} for r in rows if r.get("form") in ANNUAL_FORMS]
        if rows:
            keep[tag] = rows
    return {"cik": facts_json.get("cik"), "name": facts_json.get("entityName"), "facts": keep}


def fetch_one(client: httpx.Client, limiter: RateLimiter, cik: int) -> str:
    """Fetch one company and write data/xbrl/{cik}.json. Returns "ok", "missing" or "error"."""
    for attempt in range(5):
        limiter.wait()
        try:
            r = client.get(URL.format(cik=cik))
        except httpx.HTTPError:
            r = None  # network blip: treat like a 5xx and retry
        if r is not None and r.status_code == 200:
            (XBRL_DIR / f"{cik}.json").write_text(json.dumps(slim(r.json())))
            return "ok"
        if r is not None and r.status_code == 404:
            # No XBRL for this company. Write a marker so a rerun doesn't ask again.
            (XBRL_DIR / f"{cik}.json").write_text(json.dumps({"cik": cik, "missing": True}))
            return "missing"
        if r is not None and r.status_code not in (429, 500, 502, 503, 504):
            print(f"  cik {cik}: HTTP {r.status_code}, not retrying")
            return "error"
        time.sleep(2 ** attempt)  # 1, 2, 4, 8 s: give the SEC room to recover
    return "error"


def fetch_all(ciks: list[int], workers: int, per_second: float) -> None:
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        sys.exit('SEC_USER_AGENT is not set. The SEC requires "Your Name you@example.com".\n'
                 'Run: SEC_USER_AGENT="Your Name you@example.com" uv run python '
                 'extraction/fetch_xbrl.py')
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in ciks if not (XBRL_DIR / f"{c}.json").exists()]  # resumable
    print(f"{len(ciks)} companies, {len(ciks) - len(todo)} already cached, fetching {len(todo)}")
    limiter = RateLimiter(per_second)
    counts = {"ok": 0, "missing": 0, "error": 0}
    t0 = time.perf_counter()
    with httpx.Client(headers={"User-Agent": agent}, timeout=30) as client, \
            ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_one, client, limiter, c) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)}  {counts}  eta {(len(todo) - i) / rate:.0f}s")


# ---------- matching ----------

def days(row: dict) -> int:
    if not row.get("start"):
        return 0
    return (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days


def is_annual(row: dict) -> bool:
    return 330 <= days(row) <= 400


def pick_accession(facts: dict, year: int) -> dict | None:
    """Steps 1-3: find the one 10-K whose facts say fiscal year == `year`."""
    cands = {}
    for rows in facts.values():
        for r in rows:
            if r["fp"] == "FY" and r["fy"] == year:
                cands[r["accn"]] = r
    if not cands:
        return None
    # Original 10-K before amendments, then earliest filed.
    return min(cands.values(), key=lambda r: (r["form"] != "10-K", r["filed"]))


def value_in_filing(rows: list[dict], accn: str, flow: bool) -> dict | None:
    """Step 4: the latest-period value inside one filing (annual periods for flow items)."""
    rows = [r for r in rows if r["accn"] == accn and (is_annual(r) or not flow)]
    return max(rows, key=lambda r: r["end"]) if rows else None


def prior_value(rows: list[dict], accn: str, current_end: str) -> float | None:
    """Step 6: same filing, annual period ending ~1 year before the current one."""
    end = date.fromisoformat(current_end)
    for r in rows:
        gap = (end - date.fromisoformat(r["end"])).days
        if r["accn"] == accn and is_annual(r) and 350 <= gap <= 380:
            return r["val"]
    return None


def match_filing(company: dict, year: int) -> dict:
    """XBRL truth for one 10-K: values, which tag, accession, period end, and the rule used."""
    out = {"match_rule": "no_xbrl"}
    facts = company.get("facts") or {}
    picked = pick_accession(facts, year)
    if picked is None:
        return out if company.get("missing") or not facts else {"match_rule": "no_fy_match"}
    accn = picked["accn"]
    out = {"match_rule": "form10K_fy_FY", "accn": accn, "form": picked["form"],
           "filed": picked["filed"]}
    for tag in REVENUE_TAGS:  # step 5: first revenue tag with a value wins
        hit = value_in_filing(facts.get(tag, []), accn, flow=True)
        if hit:
            out |= {"revenue": hit["val"], "revenue_tag": tag, "period_end": hit["end"],
                    "revenue_prior": prior_value(facts[tag], accn, hit["end"])}
            break
    for field, tag in OTHER_TAGS.items():
        hit = value_in_filing(facts.get(tag, []), accn, flow=field in FLOW_FIELDS)
        if hit:
            out[field] = hit["val"]
            out.setdefault("period_end", hit["end"])
    return out


def load_company(cik: int) -> dict:
    path = XBRL_DIR / f"{cik}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def truth_table(filings: pd.DataFrame) -> pd.DataFrame:
    """Match every filing (cik, year) to its XBRL truth. Reads only the local cache."""
    cache = {}
    rows = []
    for cik, year in zip(filings["cik"], filings["year"]):
        if cik not in cache:
            cache[cik] = load_company(cik)
        rows.append({"cik": cik, "year": year, **match_filing(cache[cik], int(year))})
    return pd.DataFrame(rows)


def report(filings: pd.DataFrame) -> None:
    """Print the match rate and the filing-year vs fiscal-year check."""
    t = truth_table(filings)
    print(f"\nmatch rules over {len(t)} filings:")
    print(t["match_rule"].value_counts().to_string())
    m = t[t["match_rule"] == "form10K_fy_FY"]
    for f in ["revenue", "net_income", "total_assets", "revenue_prior"]:
        print(f"  {f:14s} found for {m[f].notna().mean():.1%} of matched filings")
    end_year = pd.to_datetime(m["period_end"]).dt.year
    filed_year = pd.to_datetime(m["filed"]).dt.year
    print(f"  period ends in `year`: {(end_year == m['year']).mean():.1%}   "
          f"filed in `year`: {(filed_year == m['year']).mean():.1%}   "
          f"filed in `year`+1: {(filed_year == m['year'] + 1).mean():.1%}")
    print("  revenue tag used:\n" + m["revenue_tag"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                        help="whose companies to fetch; 'all' also feeds the sector medians")
    parser.add_argument("--limit", type=int, help="only the first N companies (smoke test)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-second", type=float, default=8.0, help="SEC allows 10; stay under")
    parser.add_argument("--report-only", action="store_true", help="skip fetching")
    args = parser.parse_args()

    filings = load_companies()
    if args.split != "all":
        filings = filings[filings["split"] == args.split]
    ciks = sorted(int(c) for c in filings["cik"].unique())[: args.limit]
    if not args.report_only:
        fetch_all(ciks, args.workers, args.per_second)
    report(filings[filings["cik"].isin(ciks)])
