"""The analyze step: ratios, sector outlier flags, and a grounded analyst note.

Split in two on purpose:
  - Code does all the arithmetic (ratios, medians, z-scores). LLMs are unreliable
    at arithmetic; code is exact and testable.
  - The LLM only turns those finished numbers into 2-3 readable sentences, and a
    code check then confirms it did not invent or change any number.

The outlier flag is a simple statistical model, not a judgment: for each sector we
take the median and the spread (MAD) of each ratio over the XBRL values of the
train + val companies, and call a filing an outlier when its robust z-score is
above 3.5 in size (the Iglewicz-Hoaglin rule of thumb).
"""

import re
import statistics

import pandas as pd
from common import load_companies
from fetch_xbrl import truth_table

RATIOS = ["net_margin", "return_on_assets", "revenue_growth"]
Z_LIMIT = 3.5
MIN_PEERS = 10  # fewer reference filings than this: no median, no flag


# ---------- ratios ----------

def compute_ratios(values: dict, prior_revenue: float | None) -> dict:
    """Net margin, return on assets, revenue growth. None when an input is missing."""
    rev, ni, assets = values.get("revenue"), values.get("net_income"), values.get("total_assets")
    ok = lambda x: x is not None and not pd.isna(x)
    return {
        "net_margin": ni / rev if ok(ni) and ok(rev) and rev > 0 else None,
        "return_on_assets": ni / assets if ok(ni) and ok(assets) and assets > 0 else None,
        "revenue_growth": rev / prior_revenue - 1
        if ok(rev) and ok(prior_revenue) and prior_revenue > 0 else None,
    }


# ---------- robust z-scores ----------

def mad(xs: list[float]) -> float:
    """Median absolute deviation: the typical distance from the median.
    Like a standard deviation, but one wild value can't blow it up."""
    m = statistics.median(xs)
    return statistics.median(abs(x - m) for x in xs)


def robust_z(x: float | None, median: float, spread: float) -> float | None:
    """How unusual is x? 0.6745 rescales the MAD so the score reads like a normal z-score."""
    if x is None or not spread:
        return None
    return 0.6745 * (x - median) / spread


def sector_reference() -> dict:
    """{sector: {ratio: {"median", "mad", "n"}}} from XBRL truth of train + val filings.

    Test companies are left out, so a filing is never compared against itself.
    """
    filings = load_companies()
    filings = filings[filings["split"].isin(["train", "val"])][["cik", "year", "sector"]]
    t = truth_table(filings).merge(filings, on=["cik", "year"])
    rows = [dict(sector=r.sector, **compute_ratios(r._asdict(), r.revenue_prior))
            for r in t.reindex(columns=["sector", "revenue", "net_income", "total_assets",
                                        "revenue_prior"]).itertuples()]
    df = pd.DataFrame(rows)
    ref = {}
    for sector, g in df.groupby("sector"):
        ref[sector] = {}
        for ratio in RATIOS:
            xs = g[ratio].dropna().tolist()
            if len(xs) >= MIN_PEERS:
                ref[sector][ratio] = {"median": statistics.median(xs), "mad": mad(xs), "n": len(xs)}
    return ref


def flag_outliers(ratios: dict, sector_ref: dict) -> dict:
    """For each ratio: its value, the sector median, the robust z, and outlier yes/no."""
    flags = {}
    for ratio, x in ratios.items():
        ref = sector_ref.get(ratio)
        if x is None or ref is None:
            continue
        z = robust_z(x, ref["median"], ref["mad"])
        flags[ratio] = {"value": x, "sector_median": ref["median"], "z": z,
                        "outlier": z is not None and abs(z) > Z_LIMIT}
    return flags


# ---------- the note's facts, and the grounding check ----------

def fmt_money(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    for size, word in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if x >= size:
            return f"{sign}${x / size:.1f} {word}"
    return f"{sign}${x:.0f}"


def note_facts(name: str, sector: str, year: int, values: dict, flags: dict) -> dict:
    """Every number the note may use, already formatted. The LLM copies, never computes."""
    facts = {"company": name, "sector": sector, "fiscal_year": str(year)}
    for field in ("revenue", "net_income", "total_assets"):
        if values.get(field) is not None:
            facts[field] = fmt_money(values[field])
    for ratio, f in flags.items():
        facts[ratio] = f"{f['value'] * 100:.1f}%"
        facts[f"sector_median_{ratio}"] = f"{f['sector_median'] * 100:.1f}%"
        if f["z"] is not None:
            facts[f"{ratio}_robust_z"] = f"{f['z']:.1f}"
        facts[f"{ratio}_is_outlier"] = "yes" if f["outlier"] else "no"
    facts["outlier_rule"] = f"|robust z| above {Z_LIMIT}"
    return facts


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def ungrounded_numbers(note: str, facts: dict) -> list[str]:
    """Numbers in the note that are not in the facts. Empty list = fully grounded.

    Signs are ignored ("a loss of $4.2 million" for -$4.2 million is fine).
    """
    allowed = numbers_in(" ".join(str(v) for v in facts.values()))
    return [n for n in NUMBER.findall(note) if float(n.replace(",", "")) not in allowed]


NOTE_PROMPT = """You write short notes for a private-equity analyst.
Write 2-3 sentences about the company's fiscal year using ONLY the facts given.
Copy every number exactly as written in the facts. Do not calculate, round or
add any other number. If a ratio is flagged as an outlier, say so and name the
sector median. Plain text, no headings, no bullet points."""
