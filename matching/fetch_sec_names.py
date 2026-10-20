"""Fixture: download each company's current and former names from SEC EDGAR.

Run:    SEC_USER_AGENT="Your Name your@email" uv run python -m matching.fetch_sec_names
Smoke:  SEC_USER_AGENT="Your Name your@email" uv run python -m matching.fetch_sec_names --limit 20

Writes two caches into data/ (git-ignored):
- data/sec_names.jsonl: one line per CIK in data/companies.csv, with
  cik, name, formerNames (name, from, to), tickers, exchanges.
- data/company_tickers.json: SEC's list of every listed filer (the wider universe).

The script is resumable: CIKs already in sec_names.jsonl are skipped, so a
crash or Ctrl-C loses nothing. Rerun the same command to finish.
"""

import argparse
import json
import os
import sys
import time

import httpx
import pandas as pd

from matching.common import DATA, SEC_NAMES, SEC_TICKERS

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
MIN_GAP_S = 0.13  # SEC's fair-access limit is 10 requests/s; stay under 8 to be polite
KEEP = ["cik", "name", "formerNames", "tickers", "exchanges"]

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="only the first N missing CIKs (smoke test)")
args = parser.parse_args()

# SEC blocks requests without a User-Agent that names a contact. Read it from the
# environment so no name or email ever lands in the code.
user_agent = os.environ.get("SEC_USER_AGENT")
if not user_agent:
    sys.exit('Set SEC_USER_AGENT first, e.g. SEC_USER_AGENT="Your Name your@email". '
             "SEC requires a contact in the User-Agent header.")

client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30)
last_call = 0.0


def get(url):
    """GET with a rate limit and retries. Returns parsed JSON, or None for 404."""
    global last_call
    for attempt in range(5):
        wait = MIN_GAP_S - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)  # never faster than one call every MIN_GAP_S seconds
        last_call = time.monotonic()
        try:
            r = client.get(url)
            if r.status_code == 404:
                return None  # no such CIK: retrying will not help
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request,
                                            response=r)
            r.raise_for_status()  # other 4xx (403 = bad User-Agent): fail loudly
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", 0)
            if 400 <= status < 500 and status != 429:
                raise
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 ** attempt)  # back off 1, 2, 4, 8 s


# 1. The wider universe: one small file, refetched each run.
SEC_TICKERS.write_text(json.dumps(get(TICKERS)))
print(f"saved {SEC_TICKERS} ({len(json.loads(SEC_TICKERS.read_text()))} listed filers)")

# 2. One submissions file per company in our 10-K set, skipping ones already cached.
ciks = sorted(pd.read_csv(DATA / "companies.csv")["cik"].unique())
done = set()
if SEC_NAMES.exists():
    done = {json.loads(line)["cik"] for line in SEC_NAMES.read_text().splitlines()}
todo = [c for c in ciks if c not in done]
if args.limit:
    todo = todo[: args.limit]
print(f"{len(ciks)} CIKs, {len(done)} cached, fetching {len(todo)}")

missing = []
with SEC_NAMES.open("a") as f:
    for i, cik in enumerate(todo, 1):
        sub = get(SUBMISSIONS.format(cik=int(cik)))
        if sub is None:
            missing.append(int(cik))
        else:
            sub["cik"] = int(cik)
            f.write(json.dumps({k: sub.get(k, []) for k in KEEP}) + "\n")
            f.flush()  # each line lands on disk at once, which makes the script resumable
        if i % 250 == 0:
            print(f"  {i}/{len(todo)} fetched")

total = len(SEC_NAMES.read_text().splitlines())
print(f"done: {total} CIKs cached in {SEC_NAMES}; {len(missing)} not found {missing[:10]}")
