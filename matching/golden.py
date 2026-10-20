"""Build the golden set: queries with a known correct CIK.

Run:  uv run python -m matching.golden
Needs data/sec_names.jsonl first (python -m matching.fetch_sec_names).
Writes data/match_golden.csv and prints counts by type and split.

Query types:
- former_name     a name SEC says the company used before (the real "renamed" case)
- former_variant  a former name, also messed up by one of the edits below
- suffix, the, ampersand, abbreviation, case_punct, typo
                  realistic edits of the CURRENT name, the way a GP report writes it
- ticker          the stock ticker

Queries are split into dev and test BY COMPANY (seed 14, 50/50): every query
about one CIK lands on the same side, so tuning on dev never sees a test company.
"""

import json
import random
import re

import pandas as pd

from matching.common import GOLDEN, SEC_NAMES, load_universe
from matching.matcher import build_aliases, normalize

SEED = 14
KINDS = ["suffix", "the", "ampersand", "abbreviation", "case_punct", "typo"]

# Legal suffix -> what a report might write instead ("" = dropped).
SUFFIX_SWAPS = {
    "inc": ["Incorporated", "Corp", ""], "incorporated": ["Inc", ""],
    "corp": ["Corporation", "Inc", ""], "corporation": ["Corp", "Inc", ""],
    "co": ["Company", ""], "company": ["Co", ""], "ltd": ["Limited", ""],
    "limited": ["Ltd", ""], "llc": [""], "lp": [""], "plc": [""],
}
# Common abbreviations, both directions.
ABBREVIATIONS = {
    "international": "Intl", "intl": "International", "holdings": "Hldgs",
    "hldgs": "Holdings", "manufacturing": "Mfg", "services": "Svcs",
    "national": "Natl", "group": "Grp", "systems": "Sys",
}


def clean_display(name: str) -> str:
    """SEC adds tags like /DE/ that no GP report would copy. Remove them first."""
    return re.sub(r"\s*[/\\][A-Za-z]{2,5}([/\\]|$)", " ", name).strip()


def make_variant(name: str, kind: str, rng: random.Random) -> str | None:
    """One realistic edit of a name, or None if this edit does not apply."""
    words = name.split()
    match kind:
        case "suffix":  # "Acme Corp" -> "Acme Corporation" / "Acme Inc" / "Acme"
            last = words[-1].strip(".,").lower() if len(words) > 1 else ""
            if last not in SUFFIX_SWAPS:
                return None
            new = rng.choice(SUFFIX_SWAPS[last])
            out = " ".join(words[:-1]).rstrip(",") + (f" {new}" if new else "")
        case "the":  # SEC drops a leading "The"; reports often keep it. Add it or drop it.
            out = " ".join(words[1:]) if words[0].lower() == "the" else "The " + name
        case "ampersand":
            if " & " in name:  # only a spaced "&": nobody writes "H&R" as "HandR"
                out = name.replace(" & ", " and ")
            elif re.search(r"\band\b", name, re.IGNORECASE):
                out = re.sub(r"\band\b", "&", name, count=1, flags=re.IGNORECASE)
            else:
                return None
        case "abbreviation":
            hits = [i for i, w in enumerate(words) if w.strip(".,").lower() in ABBREVIATIONS]
            if not hits:
                return None
            i = rng.choice(hits)
            words[i] = ABBREVIATIONS[words[i].strip(".,").lower()]
            out = " ".join(words)
        case "case_punct":  # "AAR CORP" -> "Aar Corp"; "Apple, Inc." -> "APPLE INC"
            out = re.sub(r"[.,]", "", name)
            out = out.title() if name.isupper() else out.upper()
        case "typo":
            out = typo(name, rng)
        case _:
            raise ValueError(kind)
    return out if out and out != name else None


def typo(name: str, rng: random.Random) -> str | None:
    """One character wrong: a swap, a deletion or a doubled letter in a word of 4+ letters."""
    spots = [m for m in re.finditer(r"[A-Za-z]{4,}", name)]
    if not spots:
        return None
    m = rng.choice(spots)
    w = m.group()
    i = rng.randrange(1, len(w) - 1)  # keep the first letter: people rarely get it wrong
    match rng.choice(["swap", "delete", "double"]):
        case "swap":
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        case "delete":
            w = w[:i] + w[i + 1:]
        case "double":
            w = w[:i] + w[i] + w[i:]
    return name[: m.start()] + w + name[m.end():]


def build_rows(records: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    for r in records:
        current = clean_display(r["name"])
        seen = {normalize(current)}
        for f in r["formerNames"]:
            former = clean_display(f["name"])
            if normalize(former) in seen:
                continue  # only a case or tag change: not a real rename
            seen.add(normalize(former))
            dates = {"valid_from": (f.get("from") or "")[:10],
                     "valid_to": (f.get("to") or "")[:10]}
            rows.append({"query": former, "cik": r["cik"], "type": "former_name", **dates})
            edits = [v for k in KINDS if (v := make_variant(former, k, rng))]
            if edits:
                rows.append({"query": rng.choice(edits), "cik": r["cik"],
                             "type": "former_variant", **dates})
        # Current name: always a typo, plus one other edit that applies.
        options = [(k, v) for k in KINDS if k != "typo" and (v := make_variant(current, k, rng))]
        picks = [("typo", make_variant(current, "typo", rng))]
        if options:  # rare edits first, so each type has enough queries to measure
            rare = [o for o in options if o[0] in ("ampersand", "abbreviation")]
            picks.append(rng.choice(rare or options))
        for kind, v in picks:
            if v:
                rows.append({"query": v, "cik": r["cik"], "type": kind})
        if r["tickers"]:
            rows.append({"query": r["tickers"][0], "cik": r["cik"], "type": "ticker"})
    return rows


def main():
    rng = random.Random(SEED)
    records = [json.loads(line) for line in SEC_NAMES.read_text().splitlines()]
    df = pd.DataFrame(build_rows(records, rng)).drop_duplicates(["query", "cik"])

    # How many different companies share this query's normalized name anywhere in the
    # universe? >1 means the query is ambiguous even for a perfect matcher.
    owners = {}
    for a in build_aliases(load_universe()):
        owners.setdefault(a.norm, set()).add(a.cik)
    df["n_owners"] = [len(owners.get(normalize(q), ())) for q in df["query"]]

    ciks = sorted(df["cik"].unique())
    random.Random(SEED).shuffle(ciks)
    dev = set(ciks[: len(ciks) // 2])
    df["split"] = ["dev" if c in dev else "test" for c in df["cik"]]
    df.to_csv(GOLDEN, index=False)

    print(f"saved {GOLDEN}: {len(df)} queries about {df['cik'].nunique()} companies")
    print(df.pivot_table(index="type", columns="split", values="cik", aggfunc="count",
                         margins=True, fill_value=0))
    print(f"\nqueries whose normalized form belongs to 2+ companies: "
          f"{(df['n_owners'] > 1).sum()}")
    print("\nexamples:")
    print(df.groupby("type").head(2)[["type", "query", "cik"]].to_string(index=False))


if __name__ == "__main__":
    main()
