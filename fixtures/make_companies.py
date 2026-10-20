"""Fixture builder: public 10-K filings, 2016-2020.

Source text: EDGAR-CORPUS (Hugging Face eloukas/edgar-corpus, Apache 2.0), test and
validate files per year, downloaded to data/raw/edgar/.
Labels: each company's SIC code from the SEC submissions API, mapped to a sector.

Steps (run in order):
  uv run python fixtures/make_companies.py download  # ~1.8 GB from Hugging Face, once
  uv run python fixtures/make_companies.py texts   # local only
  SEC_USER_AGENT="Name contact@example.com" \
  uv run python fixtures/make_companies.py sic     # ~1 request per company, resumable
  uv run python fixtures/make_companies.py merge   # writes data/companies.csv

Outputs:
  data/companies_text.jsonl  one row per filing: cik, year, business, risk_factors (full text)
  data/sic_cache.json        cik -> {name, sic, sic_description}
  data/companies.csv         one row per filing: cik, year, name, sic, sic_description, sector, business
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw" / "edgar"
TEXTS = DATA / "companies_text.jsonl"
SIC_CACHE = DATA / "sic_cache.json"
OUT = DATA / "companies.csv"

MIN_BUSINESS_CHARS = 2000   # shorter sections are usually "see exhibit" stubs
BUSINESS_CHARS = 4000       # the classifier reads the opening of Item 1, where companies say what they do

# SIC code ranges -> investor-style sectors (roughly GICS). First match wins, so the
# narrow ranges come before the broad ones. Codes not listed become "other" and are
# dropped in merge. This table is the labeling guide: change it and rerun merge.
SECTOR_RANGES = [
    ((6770, 6770), "drop"),                   # blank-check shells (SPACs): no real business yet
    ((2833, 2836), "healthcare"),             # pharma, biotech
    ((3841, 3851), "healthcare"),             # medical devices, instruments
    ((8000, 8099), "healthcare"),             # hospitals, clinics, health services
    ((8731, 8731), "healthcare"),             # commercial research (mostly biotech in EDGAR)
    ((3570, 3579), "technology"),             # computers, storage
    ((3660, 3679), "technology"),             # communications equipment, semiconductors
    ((3825, 3825), "technology"),             # test equipment for electronics
    ((7370, 7379), "technology"),             # software, IT services
    ((4800, 4899), "communications_media"),   # telecom, cable, broadcasting
    ((2700, 2799), "communications_media"),   # publishing
    ((7810, 7899), "communications_media"),   # film, entertainment
    ((1300, 1399), "energy"),                 # oil and gas extraction, services
    ((2900, 2999), "energy"),                 # refining
    ((4610, 4619), "energy"),                 # pipelines
    ((4900, 4999), "utilities"),              # electric, gas, water
    ((6500, 6599), "real_estate"),
    ((6798, 6798), "real_estate"),            # REITs
    ((6000, 6799), "financials"),             # banks, lenders, brokers, insurance, holding offices
    ((1000, 1099), "materials"),              # metal mining
    ((1400, 1499), "materials"),              # nonmetal mining
    ((2400, 2499), "materials"),              # lumber
    ((2600, 2699), "materials"),              # paper, packaging
    ((2800, 2899), "materials"),              # chemicals
    ((3300, 3399), "materials"),              # primary metals
    ((2000, 2399), "consumer"),               # food, beverages, tobacco, textiles, apparel
    ((2500, 2599), "consumer"),               # furniture
    ((3940, 3949), "consumer"),               # toys, sporting goods
    ((5200, 5999), "consumer"),               # retail, restaurants
    ((7000, 7099), "consumer"),               # hotels
    ((100, 999), "consumer"),                 # agriculture (food staples)
    ((3630, 3639), "consumer"),               # household appliances
    ((3900, 3999), "consumer"),               # jewelry, misc. consumer manufacturing
    ((7200, 7299), "consumer"),               # personal services
    ((7900, 7999), "consumer"),               # amusement, recreation, gaming
    ((8200, 8299), "consumer"),               # education
    ((7310, 7319), "communications_media"),   # advertising
    ((3826, 3826), "healthcare"),             # lab analytical instruments (life-science tools)
    ((3000, 3299), "materials"),              # rubber, plastics, glass, stone
    ((1200, 1299), "energy"),                 # coal
    ((1500, 1799), "industrials"),            # construction
    ((3400, 3569), "industrials"),            # fabricated metal, machinery
    ((3580, 3699), "industrials"),            # electrical equipment
    ((3700, 3799), "industrials"),            # aerospace, autos, transport equipment
    ((3800, 3839), "industrials"),            # industrial measuring and control instruments
    ((4000, 4799), "industrials"),            # rail, trucking, shipping, airlines, logistics
    ((5000, 5199), "industrials"),            # wholesale distribution
    ((7300, 7399), "industrials"),            # business services, staffing, equipment rental
    ((7500, 7549), "industrials"),            # vehicle rental and leasing
    ((8700, 8799), "industrials"),            # engineering, consulting, management services
]


def sector_for(sic: int | None) -> str:
    if sic is None:
        return "other"
    for (lo, hi), sector in SECTOR_RANGES:
        if lo <= sic <= hi:
            return sector
    return "other"


HF_URL = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{year}/{split}.jsonl"


def download() -> None:
    """Fetch the 2016-2020 test and validate files (the train files are 1.5 GB per year)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for year in range(2016, 2021):
        for split in ("test", "validate"):
            dest = RAW / f"{year}_{split}.jsonl"
            if dest.exists():
                continue
            print(f"downloading {year}/{split}.jsonl")
            urllib.request.urlretrieve(HF_URL.format(year=year, split=split), dest)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_texts() -> None:
    seen = set()
    rows = 0
    with TEXTS.open("w") as out:
        for path in sorted(RAW.glob("*.jsonl")):
            for line in path.open():
                d = json.loads(line)
                business = clean(d.get("section_1"))
                key = (d["cik"], d["year"])
                if len(business) < MIN_BUSINESS_CHARS or key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({
                    "cik": d["cik"],
                    "year": int(d["year"]),
                    "business": business,
                    "risk_factors": clean(d.get("section_1A")),
                }) + "\n")
                rows += 1
    print(f"wrote {rows} filings to {TEXTS}")


def fetch_sic() -> None:
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        sys.exit('Set SEC_USER_AGENT="Your Name your@email" first (the SEC requires a contact).')
    cache = json.loads(SIC_CACHE.read_text()) if SIC_CACHE.exists() else {}
    ciks = sorted({json.loads(line)["cik"] for line in TEXTS.open()})
    todo = [c for c in ciks if c not in cache]
    print(f"{len(ciks)} companies, {len(todo)} to fetch")
    for i, cik in enumerate(todo, 1):
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        req = urllib.request.Request(url, headers={"User-Agent": agent})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                d = json.load(resp)
            sic = d.get("sic")
            cache[cik] = {
                "name": d.get("name", ""),
                "sic": int(sic) if sic else None,
                "sic_description": d.get("sicDescription", ""),
            }
        except Exception as e:  # keep going; a rerun retries the misses
            print(f"  {cik}: {e}")
        time.sleep(0.15)  # SEC fair-access limit is 10 requests per second
        if i % 200 == 0:
            SIC_CACHE.write_text(json.dumps(cache))
            print(f"  {i}/{len(todo)}")
    SIC_CACHE.write_text(json.dumps(cache))
    print(f"cached {len(cache)} companies in {SIC_CACHE}")


def merge() -> None:
    cache = json.loads(SIC_CACHE.read_text())
    counts: dict[str, int] = {}
    kept = 0
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cik", "year", "name", "sic", "sic_description", "sector", "business"])
        for line in TEXTS.open():
            d = json.loads(line)
            info = cache.get(d["cik"])
            sector = sector_for(info["sic"]) if info else "other"
            counts[sector] = counts.get(sector, 0) + 1
            if sector in ("other", "drop"):
                continue
            w.writerow([d["cik"], d["year"], info["name"], info["sic"], info["sic_description"],
                        sector, d["business"][:BUSINESS_CHARS]])
            kept += 1
    for sector, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sector:22s} {n}")
    print(f"wrote {kept} filings to {OUT}")


if __name__ == "__main__":
    steps = {"download": download, "texts": build_texts, "sic": fetch_sic, "merge": merge}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit(f"usage: make_companies.py {{{'|'.join(steps)}}}")
    steps[sys.argv[1]]()
