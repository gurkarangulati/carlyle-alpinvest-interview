"""Bronze: land the source documents for every company the fund reports name.

Run:  uv run python -m lake.ingest

Steps
  1. Read the fund report PDFs (parse_pdf.py) and resolve each reported name
     to an SEC company with the matching module, as of the report date.
  2. For each resolved company, copy its fiscal-2019 and fiscal-2018 10-K
     sections (1, 1A, 7, plus 8 and 15 for the extraction graph) from
     data/raw/edgar into data/lake/bronze/filings/{cik}_{year}.json.
  3. Copy its cached SEC XBRL facts to data/lake/bronze/xbrl/{cik}.json.
  4. Write its SEC registry entry (current name, former names, SIC, sector)
     to data/lake/bronze/reference/entities.jsonl: the raw input of the entity master.

Bronze is a copy, not a transformation: if a later step has a bug, we rebuild
silver and gold from these files without going back to the sources.
The company list comes from the documents, not from the truth files: in real
life the LP only knows what the GP reports say.
"""

import json
import shutil

import pandas as pd

from lake.common import (
    AS_OF,
    BRONZE_XBRL,
    DATA,
    FILINGS,
    FISCAL_YEARS,
    FUND_REPORTS,
    REFERENCE,
    SECTIONS,
    XBRL_CACHE,
    load_matcher,
    resolve,
    use_extraction_modules,
)
from lake.parse_pdf import parse_fund_pdf


def resolved_ciks() -> dict[int, list[str]]:
    """{cik: [reported names]} for every holding the matcher resolved."""
    matcher = load_matcher()
    found = {}
    for pdf in sorted(FUND_REPORTS.glob("*.pdf")):
        parsed = parse_fund_pdf(pdf)
        names = [h["reported_name"] for h in parsed["holdings"]]
        for name, r in zip(names, resolve(matcher, names, parsed["fund"]["as_of"] or AS_OF)):
            if r["match_decision"] == "match":
                found.setdefault(r["cik"], []).append(name)
            else:
                print(f"  abstained on {name!r} (best score {r['match_score']}): no documents pulled")
    return found


def copy_filings(ciks: set[int]) -> int:
    """Stream the raw 10-K files once and write the sections we keep. Returns files written."""
    use_extraction_modules()
    from common import load_texts  # extraction/common.py: streams the big jsonl files

    keys = {(cik, year) for cik in ciks for year in FISCAL_YEARS}
    texts = load_texts(keys, sections=SECTIONS)
    FILINGS.mkdir(parents=True, exist_ok=True)
    for (cik, year), sections in sorted(texts.items()):
        doc = {"cik": cik, "fiscal_year": year, "source": f"EDGAR-CORPUS {year} 10-K",
               "sections": sections}
        (FILINGS / f"{cik}_{year}.json").write_text(json.dumps(doc))
    missing = sorted(keys - set(texts))
    if missing:
        print(f"  no 10-K in the corpus for {len(missing)} (cik, year): {missing[:8]}")
    return len(texts)


def copy_xbrl(ciks: set[int]) -> int:
    BRONZE_XBRL.mkdir(parents=True, exist_ok=True)
    n = 0
    for cik in sorted(ciks):
        src = XBRL_CACHE / f"{cik}.json"
        if src.exists():
            shutil.copyfile(src, BRONZE_XBRL / f"{cik}.json")
            n += 1
    return n


def write_reference(ciks: set[int]) -> None:
    """SEC name history + the latest SIC/sector we have, one line per company."""
    names = {}
    for line in (DATA / "sec_names.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["cik"] in ciks:
            names[r["cik"]] = r
    tickers = {int(t["cik_str"]): t["title"]
               for t in json.loads((DATA / "company_tickers.json").read_text()).values()}
    meta = pd.read_csv(DATA / "companies.csv", usecols=["cik", "year", "sic", "sector"])
    meta = meta[meta["cik"].isin(ciks)].sort_values("year").groupby("cik").last()
    REFERENCE.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE / "entities.jsonl", "w") as f:
        for cik in sorted(ciks):
            r = names.get(cik, {"name": tickers.get(cik, ""), "formerNames": []})
            row = {"cik": cik, "name": r["name"], "former_names": r["formerNames"],
                   "sic": int(meta.loc[cik, "sic"]) if cik in meta.index else None,
                   "sector": meta.loc[cik, "sector"] if cik in meta.index else None}
            f.write(json.dumps(row) + "\n")


def main():
    found = resolved_ciks()
    ciks = set(found)
    print(f"{len(ciks)} companies resolved from {len(list(FUND_REPORTS.glob('*.pdf')))} "
          "fund reports")
    print("copying 10-K sections (streams the raw 10-K files once)...")
    n_filings = copy_filings(ciks)
    n_xbrl = copy_xbrl(ciks)
    write_reference(ciks)
    print(f"bronze: {n_filings} filings, {n_xbrl} XBRL files, {len(ciks)} reference entities")


if __name__ == "__main__":
    main()
