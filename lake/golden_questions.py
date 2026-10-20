"""~55 analyst questions with expected answers, for grading an agent that queries the lake.

Run:  uv run python -m lake.golden_questions
Out:  data/lake/golden/questions.jsonl

The expected answers come from the ANSWER KEY, never from the lake tables:
  - fund_truth.csv / funds_truth.csv   what funds.py really put in the reports
  - SEC XBRL facts (data/xbrl)         the financial numbers
  - SEC name history (sec_names.jsonl) renames
  - the raw 10-K text (data/raw/edgar) risk-factor passages
If the lake has a bug (a wrong match, a misread PDF), an agent reading the lake
gets these questions wrong, which is exactly what a golden set should catch.

Fields per line
  id, type, question
  expected        the answer: a number (USD, or percent where the question says so),
                  a list (a set: CIKs or fund ids), an entity dict, or text
  answer_kind     number | set | text | entity | unanswerable
  tolerance       numbers only: {"rel": 0.005} = within 0.5%, {"abs": 0.1} = within 0.1
  expected_sources  {"tables": [...], "chunk_prefixes": [...], "chunk_ids": [...],
                     "keywords": [...]} where a grounded answer comes from;
                  unanswerable questions carry {"reason": ...} instead
  expected_display  human-readable version of `expected` (names instead of CIKs)
"""

import json
import random

import pandas as pd

from lake.common import AS_OF, DATA, GOLDEN, XBRL_CACHE, chunk_text, use_extraction_modules, xbrl_values
from matching.golden import clean_display
from matching.matcher import normalize

SEED = 14
REL = {"rel": 0.005}


def q(type_, question, expected, kind, tolerance=None, display=None, **sources) -> dict:
    return {"type": type_, "question": question, "expected": expected, "answer_kind": kind,
            "tolerance": tolerance, "expected_sources": sources,
            "expected_display": display if display is not None else expected}


class Key:
    """Everything the expected answers are computed from."""

    def __init__(self):
        self.h = pd.read_csv(GOLDEN / "fund_truth.csv")
        self.f = pd.read_csv(GOLDEN / "funds_truth.csv").set_index("fund_id")
        self.sec = {}
        for line in (DATA / "sec_names.jsonl").read_text().splitlines():
            r = json.loads(line)
            self.sec[r["cik"]] = r
        self.x = {}
        for cik in self.h["cik"].unique():
            facts = json.loads((XBRL_CACHE / f"{cik}.json").read_text())
            self.x[int(cik)] = {y: xbrl_values(facts, y) for y in (2018, 2019)}

    def name(self, cik: int) -> str:
        return self.sec[cik]["name"]

    def fund(self, fid: str) -> str:
        return self.f.loc[fid, "fund_name"]

    def revenue(self, cik: int, year: int) -> float | None:
        v = self.x[cik][year].get("revenue")
        if v is None and year == 2018:  # comparative column of the 2019 10-K
            v = self.x[cik][2019].get("revenue_prior")
        return v

    def growth(self, cik: int) -> float | None:
        r19, r18 = self.revenue(cik, 2019), self.revenue(cik, 2018)
        return round(100 * (r19 / r18 - 1), 2) if r19 and r18 else None

    def net_income(self, cik: int) -> float | None:
        return self.x[cik][2019].get("net_income")


def entity_questions(k: Key, rng: random.Random) -> list[dict]:
    out = []
    hard = k.h[k.h["name_kind"].isin(["typo", "former_name", "former_variant", "suffix"])]
    for r in hard.sample(6, random_state=SEED).itertuples():
        out.append(q("entity_resolution",
                     f"In {k.fund(r.fund_id)}'s Q4 2019 report, which SEC-registered company is "
                     f"the holding listed as '{r.reported_name}'? Give its CIK.",
                     {"cik": int(r.cik), "name": k.name(r.cik)}, "entity",
                     tables=["holdings", "companies"]))
    # Old names no report uses: the lake must connect them through `aliases`.
    held = set(k.h["cik"])
    olds = []
    for cik in sorted(held):
        for f in k.sec[cik]["formerNames"]:
            to = (f.get("to") or "")[:10]
            if "2014-01-01" <= to < AS_OF and normalize(f["name"]) != normalize(k.name(cik)):
                olds.append((clean_display(f["name"]), cik))
    for old, cik in rng.sample(olds, min(3, len(olds))):
        funds = sorted(k.h[k.h["cik"] == cik]["fund_id"])
        out.append(q("entity_resolution",
                     f"Is {old} (under that or a later name) held by any fund? List the fund ids.",
                     funds, "set", display=[k.fund(x) for x in funds],
                     tables=["companies", "holdings"]))
    out.append(q("entity_resolution", "Is Exxon Mobil Corp held by any fund? List the fund ids.",
                 [], "set", tables=["companies", "holdings"]))
    return out


def look_through_questions(k: Key) -> list[dict]:
    out = []
    groups = k.h.groupby(["fund_id", "sector"])
    picks = [(fid, s) for (fid, s), g in groups if len(g) >= 2][:6]
    for fid, sector in picks:
        g = groups.get_group((fid, sector))
        out.append(q("look_through", f"List {k.fund(fid)}'s holdings in the {sector} sector "
                     "(as of 2019-12-31). Give CIKs.", sorted(int(c) for c in g["cik"]), "set",
                     display=sorted(k.name(c) for c in g["cik"]), tables=["holdings"]))
    shared = k.h.groupby("cik").filter(lambda g: len(g) > 1)["cik"].unique()
    for cik in sorted(shared)[:2]:
        funds = sorted(k.h[k.h["cik"] == cik]["fund_id"])
        out.append(q("look_through", f"Which funds hold {k.name(cik)}?", funds, "set",
                     display=[k.fund(x) for x in funds], tables=["holdings", "companies"]))
    return out


def aggregation_questions(k: Key) -> list[dict]:
    h, f, out = k.h, k.f, []
    for fid, sector in [("F01", "technology"), ("F02", "materials"), ("F04", "real_estate")]:
        g = h[(h["fund_id"] == fid) & (h["sector"] == sector)]
        out.append(q("aggregation", f"What is the total fair value (USD) of {k.fund(fid)}'s "
                     f"{sector} holdings?", float(g["fair_value_usd"].sum()), "number", REL,
                     tables=["holdings", "fund_exposure"]))
    for fid, sector in [("F03", "consumer"), ("F05", "financials")]:
        g = h[(h["fund_id"] == fid) & (h["sector"] == sector)]
        pct = round(100 * g["fair_value_usd"].sum() / f.loc[fid, "nav_usd"], 2)
        out.append(q("aggregation", f"What percent of {k.fund(fid)}'s NAV is in {sector} "
                     "holdings?", pct, "number", {"abs": 0.1}, tables=["fund_exposure"]))
    for fid in ("F02", "F04"):
        g = h[h["fund_id"] == fid]
        moic = round(g["fair_value_usd"].sum() / g["cost_usd"].sum(), 3)
        out.append(q("aggregation", f"What is {k.fund(fid)}'s gross multiple on its holdings "
                     "(total fair value / total cost)?", moic, "number", {"abs": 0.01},
                     tables=["holdings"]))
    out.append(q("aggregation", f"How many holdings does {k.fund('F05')} report?",
                 int((h["fund_id"] == "F05").sum()), "number", {"abs": 0}, tables=["holdings"]))
    out.append(q("aggregation", f"How much of {k.fund('F03')}'s commitment is uncalled (USD)?",
                 float(f.loc["F03", "commitment_usd"] - f.loc["F03", "called_usd"]), "number", REL,
                 tables=["funds"]))
    out.append(q("aggregation", "Across all five funds, what is the total fair value (USD) of "
                 "energy holdings?", float(h[h["sector"] == "energy"]["fair_value_usd"].sum()),
                 "number", REL, tables=["holdings"]))
    out.append(q("aggregation", "Which fund has the largest NAV?",
                 {"fund_id": f["nav_usd"].idxmax(), "fund_name": k.fund(f["nav_usd"].idxmax())},
                 "entity", tables=["funds"]))
    return out


def financial_questions(k: Key, rng: random.Random) -> list[dict]:
    out = []
    ciks = sorted(int(c) for c in k.h["cik"].unique())
    with_growth = [c for c in ciks if k.growth(c) is not None]
    for cik in rng.sample(with_growth, 4):
        out.append(q("financials", f"What was {k.name(cik)}'s revenue growth from fiscal 2018 to "
                     "fiscal 2019, in percent?", k.growth(cik), "number", {"abs": 0.1},
                     tables=["company_financials", "metrics"]))
    for cik in rng.sample([c for c in ciks if k.net_income(c) is not None], 3):
        out.append(q("financials", f"What was {k.name(cik)}'s net income for fiscal 2019 (USD)?",
                     float(k.net_income(cik)), "number", REL,
                     tables=["company_financials", "metrics"]))
    for cik in rng.sample([c for c in ciks if k.x[c][2019].get("total_assets")], 2):
        out.append(q("financials", f"What were {k.name(cik)}'s total assets at the end of fiscal "
                     "2019 (USD)?", float(k.x[cik][2019]["total_assets"]), "number", REL,
                     tables=["company_financials", "metrics"]))
    f02 = k.h[k.h["fund_id"] == "F02"]["cik"]
    top = max(f02, key=lambda c: k.revenue(c, 2019) or 0)
    out.append(q("financials", f"Which {k.fund('F02')} holding had the highest fiscal 2019 "
                 "revenue?", {"cik": int(top), "name": k.name(top)}, "entity",
                 tables=["holdings", "company_financials"]))
    return out


def cross_cutting_questions(k: Key) -> list[dict]:
    h, f, out = k.h.copy(), k.f, []
    h["loss"] = [(k.net_income(c) or 0) < 0 for c in h["cik"]]
    share = (h[h["loss"]].groupby("fund_id")["fair_value_usd"].sum() / f["nav_usd"]).fillna(0)
    top = share.idxmax()
    out.append(q("cross_cutting", "Weighting each holding by its fair value, which fund has the "
                 "largest share of its NAV in companies that reported a negative net income for "
                 "fiscal 2019?", {"fund_id": top, "fund_name": k.fund(top)}, "entity",
                 tables=["holdings", "funds", "company_financials"]))
    out.append(q("cross_cutting", f"What percent of {k.fund(top)}'s NAV is in companies with "
                 "negative fiscal-2019 net income?", round(100 * share[top], 2), "number",
                 {"abs": 0.1}, tables=["holdings", "funds", "company_financials"]))
    shared = sorted(int(c) for c in h.groupby("cik").filter(lambda g: len(g) > 1)["cik"].unique())
    out.append(q("cross_cutting", "Which companies are held by more than one fund? Give CIKs.",
                 shared, "set", display=[k.name(c) for c in shared], tables=["holdings"]))
    renamed = h[h["name_kind"].isin(["former_name", "former_variant"])]
    out.append(q("cross_cutting", "What is the total fair value (USD), across all funds, of "
                 "holdings whose company has changed its SEC name since 2019-12-31 (the report "
                 "uses the old name)?", float(renamed["fair_value_usd"].sum()), "number", REL,
                 display=sorted(k.name(c) for c in renamed["cik"]),
                 tables=["holdings", "companies"]))
    g = h[h["fund_id"] == "F01"].copy()
    g["growth"] = [k.growth(c) for c in g["cik"]]
    g = g.dropna(subset=["growth"])
    wavg = round(float((g["growth"] * g["fair_value_usd"]).sum() / g["fair_value_usd"].sum()), 2)
    out.append(q("cross_cutting", f"What is the fair-value-weighted average fiscal 2018-to-2019 "
                 f"revenue growth (percent) of {k.fund('F01')}'s holdings?", wavg, "number",
                 {"abs": 0.5}, tables=["holdings", "company_financials"]))
    tech = h[h["sector"] == "technology"].groupby("fund_id")["fair_value_usd"].sum()
    out.append(q("cross_cutting", "Which fund has the most fair value in technology holdings?",
                 {"fund_id": tech.idxmax(), "fund_name": k.fund(tech.idxmax())}, "entity",
                 tables=["fund_exposure"]))
    return out


# (cik, keyword that must appear, topic for the question)
TEXT_TOPICS = [(19617, "LIBOR", "the transition away from LIBOR"),
               (64803, "opioid", "opioids"),
               (723125, "tariff", "tariffs and trade restrictions"),
               (1326801, "privacy", "user privacy"),
               (1398987, "housing", "the housing market"),
               (1451505, "climate", "climate change")]


def text_questions(k: Key) -> list[dict]:
    use_extraction_modules()
    from common import load_texts  # extraction/common.py: reads the raw 10-K files

    texts = load_texts({(c, 2019) for c, _, _ in TEXT_TOPICS}, sections=["section_1A"])
    out = []
    for cik, kw, topic in TEXT_TOPICS:
        chunks = chunk_text(texts[(cik, 2019)]["section_1A"])
        ids = [f"10k-{cik}-2019-item1a-{i:03d}" for i, c in enumerate(chunks, 1)
               if kw.lower() in c.lower()]
        if not ids:
            continue
        out.append(q("text", f"What does {k.name(cik)}'s fiscal-2019 10-K say in its risk "
                     f"factors about {topic}? Quote the passage you rely on.",
                     f"The risk factors discuss {topic} (look for '{kw}').", "text",
                     chunk_prefixes=[f"10k-{cik}-2019-item1a"], chunk_ids=ids[:5],
                     keywords=[kw]))
    # The fund commentary names renames that happened during 2019.
    funds = sorted({r.fund_id for r in k.h.itertuples() for f in k.sec[r.cik]["formerNames"]
                    if "2019-01-01" <= (f.get("to") or "")[:10] <= AS_OF
                    and normalize(f["name"]) != normalize(k.name(r.cik))
                    and r.name_kind != "typo"})
    out.append(q("text", "Which fund reports' commentary mention a portfolio company that "
                 "filed with the SEC under a different name earlier in 2019? Give fund ids.",
                 funds, "set", display=[k.fund(x) for x in funds],
                 chunk_prefixes=[f"fund-{x}-p2" for x in funds], keywords=["filed with the SEC as"]))
    return out


def unanswerable_questions(k: Key) -> list[dict]:
    first = k.name(int(k.h["cik"].iloc[0]))
    return [
        q("unanswerable", f"What was {first}'s revenue in fiscal 2021?", "unanswerable",
          "unanswerable", reason="the lake holds fiscal 2018 and 2019 only"),
        q("unanswerable", f"What is {k.fund('F02')}'s net IRR?", "unanswerable", "unanswerable",
          reason="fund reports give NAV, commitment and called capital, not IRR"),
        q("unanswerable", f"What was {k.fund('F04')}'s NAV on 2020-06-30?", "unanswerable",
          "unanswerable", reason="only Q4 2019 reports are in the lake"),
        q("unanswerable", f"What management fee rate does {k.fund('F01')} charge?",
          "unanswerable", "unanswerable", reason="not in the reports"),
        q("unanswerable", f"What was the fair value of {k.fund('F03')}'s holdings on 2018-12-31?",
          "unanswerable", "unanswerable", reason="no 2018 fund reports"),
    ]


def main():
    rng = random.Random(SEED)
    k = Key()
    qs = (entity_questions(k, rng) + look_through_questions(k) + aggregation_questions(k)
          + financial_questions(k, rng) + cross_cutting_questions(k) + text_questions(k)
          + unanswerable_questions(k))
    counts = {}
    with open(GOLDEN / "questions.jsonl", "w") as f:
        for item in qs:
            counts[item["type"]] = counts.get(item["type"], 0) + 1
            row = {"id": f"{item['type']}-{counts[item['type']]:02d}", **item}
            f.write(json.dumps(row, default=float) + "\n")
    print(f"{len(qs)} golden questions -> {(GOLDEN / 'questions.jsonl').relative_to(DATA.parent)}")
    for t, n in counts.items():
        print(f"  {t:18s} {n}")


if __name__ == "__main__":
    main()
