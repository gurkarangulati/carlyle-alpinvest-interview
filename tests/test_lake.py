"""Tests for the data lake. No network, no LLM, no big data files: tiny made-up inputs.

Run:  uv run python -m pytest tests/test_lake.py -q
"""

import random

import duckdb
import pandas as pd
import pytest

from lake.build import (
    chunk_rows,
    companies_frame,
    extracted_rows,
    fund_frames,
    write_lake,
    xbrl_rows,
)
from lake.common import GOLD_VIEWS, SILVER_TABLES, chunk_text, sector_code, sector_label
from lake.funds import former_name_on, reported_name, write_fund_pdf
from lake.parse_pdf import parse_fund_pdf, scale_of, to_number
from lake.validate import check_nav, check_parse, check_schema, matching_scores
from matching.matcher import Matcher, normalize

AS_OF = "2019-12-31"

# ---- reported-name variants ----------------------------------------------------

RENAMED_LATER = {"cik": 1, "name": "Meta Platforms, Inc.",
                 "former_names": [{"name": "Facebook Inc", "from": "2005-05-06",
                                   "to": "2021-10-27"}]}
RENAMED_EARLIER = {"cik": 2, "name": "Valaris plc",
                   "former_names": [{"name": "Ensco plc", "from": "2010-03-18",
                                     "to": "2019-03-29"}]}
CASE_ONLY = {"cik": 3, "name": "Omnicom Group Inc.",
             "former_names": [{"name": "OMNICOM GROUP INC", "from": "2000-01-01",
                               "to": "2022-01-01"}]}


def test_former_name_used_when_valid_on_report_date():
    assert former_name_on(RENAMED_LATER, AS_OF) == "Facebook Inc"
    name, kind = reported_name(RENAMED_LATER, random.Random(0))
    assert kind in ("former_name", "former_variant")
    assert normalize(name).startswith("facebook")


def test_former_name_not_used_after_rename():
    assert former_name_on(RENAMED_EARLIER, AS_OF) is None  # Ensco ended before 2019-12-31
    assert former_name_on(CASE_ONLY, AS_OF) is None        # case change only: not a rename


def test_reported_name_is_deterministic_and_close():
    a = [reported_name(RENAMED_EARLIER, random.Random(14)) for _ in range(3)]
    assert len(set(a)) == 1  # same seed, same name
    kinds = {reported_name(RENAMED_EARLIER, random.Random(s))[1] for s in range(40)}
    assert kinds <= {"current", "suffix", "case_punct", "abbreviation", "typo"}
    assert "current" in kinds and len(kinds) >= 3


def test_no_dangling_ampersand():
    company = {"cik": 4, "name": "Arthur J. Gallagher & Co.", "former_names": []}
    for s in range(60):
        name, _ = reported_name(company, random.Random(s))
        assert not name.rstrip(" ,").endswith("&")


# ---- small helpers -------------------------------------------------------------

def test_numbers_and_scales():
    assert to_number("1,234") == 1234.0
    assert to_number("(56)") == -56.0
    assert to_number("1.25%") == 1.25
    with pytest.raises(ValueError):
        to_number("n/a")
    assert scale_of("Cost (USD 000s)") == 1e3 and scale_of("Ownership") == 1.0
    assert sector_code(sector_label("communications_media")) == "communications_media"


def test_chunks_are_stable_and_bounded():
    text = "word " * 2_000
    chunks = chunk_text(text, size=300)
    assert all(len(c) <= 300 for c in chunks)
    assert " ".join(chunks) == text.strip()
    assert chunk_text(text, size=300) == chunks


# ---- PDF round trip ------------------------------------------------------------

SUMMARY = {"fund_id": "T01", "fund_name": "Testfield Example Fund I", "vintage": 2017,
           "commitment_usd": 100_000_000.0, "called_usd": 80_000_000.0,
           "nav_usd": 95_000_000.0, "as_of": AS_OF}
HOLDINGS = [
    {"reported_name": "Facebook Inc", "sector": "technology", "cost_usd": 40_000_000.0,
     "fair_value_usd": 61_250_000.0, "ownership_pct": 0.05},
    {"reported_name": "JPMORGAN CHAE & CO", "sector": "financials", "cost_usd": 30_000_000.0,
     "fair_value_usd": 31_000_000.0, "ownership_pct": 1.5},
    {"reported_name": "Charter Communications", "sector": "communications_media",
     "cost_usd": 5_000_000.0, "fair_value_usd": 1_000.0, "ownership_pct": 2.75},
]


@pytest.fixture
def tiny_pdf(tmp_path):
    path = tmp_path / "T01.pdf"
    write_fund_pdf(SUMMARY, HOLDINGS, ["Facebook Inc reported revenue growth.",
                                       "A second paragraph."], path)
    return path


def test_pdf_round_trip(tiny_pdf):
    parsed = parse_fund_pdf(tiny_pdf)
    assert parsed["fund"] == {k: v for k, v in SUMMARY.items()}
    got = [{k: h[k] for k in HOLDINGS[0]} for h in parsed["holdings"]]
    assert got == HOLDINGS
    assert {h["source_page"] for h in parsed["holdings"]} == {3}
    assert parsed["commentary"][0]["page"] == 2
    assert "Facebook Inc reported revenue growth. A second paragraph." in \
        parsed["commentary"][0]["text"]
    assert "SYNTHETIC" not in parsed["commentary"][0]["text"]


# ---- the built tables follow the contract (tiny fixture) ------------------------

UNIVERSE = [{"cik": 1326801, "name": "Meta Platforms, Inc.", "tickers": ["META"],
             "former_names": [{"name": "Facebook Inc", "from": "2005-05-06", "to": "2021-10-27"}]},
            {"cik": 19617, "name": "JPMORGAN CHASE & CO", "tickers": ["JPM"], "former_names": []},
            {"cik": 1091667, "name": "CHARTER COMMUNICATIONS, INC. /MO/", "tickers": ["CHTR"],
             "former_names": []},
            {"cik": 1001, "name": "Facet Industries Inc", "tickers": [], "former_names": []}]


def fact(val, start, end, fy, accn="a-2019"):
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": "FY",
            "form": "10-K", "filed": f"{fy + 1}-02-01"}


XBRL = {"facts": {
    "Revenues": [fact(120.0, "2019-01-01", "2019-12-31", 2019),
                 fact(100.0, "2018-01-01", "2018-12-31", 2019)],
    "NetIncomeLoss": [fact(-5.0, "2019-01-01", "2019-12-31", 2019)],
    "Assets": [fact(900.0, None, "2019-12-31", 2019)]}}


@pytest.fixture
def tiny_lake(tmp_path, tiny_pdf):
    parsed = [parse_fund_pdf(tiny_pdf)]
    entities = [{"cik": u["cik"], "name": u["name"], "former_names": u["former_names"],
                 "sic": 7370, "sector": "technology"} for u in UNIVERSE[:3]]
    funds, holdings = fund_frames(parsed, Matcher(UNIVERSE, min_score=70, min_margin=0))
    run = {"cik": 1326801, "year": 2019, "status": "done",
           "values": {"revenue": 120.5, "net_income": float("nan"), "total_assets": None,
                      "operating_income": "7", "cash": True},
           "extracted": {"revenue_evidence": "revenue of $120.5"}}
    metrics = xbrl_rows(1326801, XBRL) + extracted_rows(run)
    filings = {(1326801, 2019): {"sections": {"section_1A": "Risk " * 400, "section_7": "MD&A"}}}
    frames = {"companies": companies_frame(entities), "funds": funds, "holdings": holdings,
              "metrics": pd.DataFrame(metrics),
              "doc_chunks": pd.DataFrame(chunk_rows(filings, parsed))}
    write_lake(frames, tmp_path)
    return tmp_path


def test_tables_match_the_contract(tiny_lake):
    con = duckdb.connect(str(tiny_lake / "lake.duckdb"), read_only=True)
    assert all(r["status"] == "pass" for r in check_schema(con))
    for view in GOLD_VIEWS:
        assert con.execute(f"SELECT count(*) FROM {view}").fetchone()[0] > 0
    for table, cols in SILVER_TABLES.items():
        pq = con.execute(f"DESCRIBE SELECT * FROM '{tiny_lake}/silver/{table}/data.parquet'")
        assert [r[0] for r in pq.fetchall()] == [c for c, _ in cols]


def test_tiny_lake_contents(tiny_lake):
    con = duckdb.connect(str(tiny_lake / "lake.duckdb"), read_only=True)
    h = con.execute("SELECT reported_name, cik, match_decision FROM holdings "
                    "ORDER BY reported_name").fetchall()
    assert ("Facebook Inc", 1326801, "match") in h          # former name, valid in 2019
    growth = con.execute("SELECT revenue_growth_pct FROM company_financials "
                         "WHERE fiscal_year = 2019").fetchone()[0]
    assert growth == 20.0                                    # 120 vs. 100 (prior column)
    ext = dict(con.execute("SELECT metric, status FROM metrics WHERE source = 'extracted'")
               .fetchall())
    # LLM outputs are validated in code: NaN, a string and a bool are not numbers.
    assert ext == {"revenue": "ok", "net_income": "missing", "total_assets": "missing",
                   "operating_income": "missing", "cash": "missing"}
    rec = con.execute("SELECT agrees_within_1pct FROM metric_reconciliation "
                      "WHERE metric = 'revenue'").fetchone()[0]
    assert rec is True                                       # 120.5 vs 120: within 1%
    ids = [r[0] for r in con.execute("SELECT chunk_id FROM doc_chunks ORDER BY 1").fetchall()]
    assert ids[0] == "10k-1326801-2019-item1a-001" and "fund-T01-p2-001" in ids


# ---- validate checks -----------------------------------------------------------

TRUTH = pd.DataFrame({"fund_id": ["F1"] * 4, "reported_name": ["A", "B", "C", "D"],
                      "cik": [1, 2, 3, 4], "name_kind": ["current"] * 4,
                      "cost_usd": [10.0] * 4, "fair_value_usd": [20.0] * 4})


def test_matching_precision_and_recall():
    lake = TRUTH[["fund_id", "reported_name"]].assign(
        cik=[1, 2, 99, None], match_decision=["match", "match", "match", "abstain"])
    s = matching_scores(lake, TRUTH)
    assert s["precision"] == pytest.approx(2 / 3)   # 3 answers, 2 right
    assert s["recall"] == pytest.approx(2 / 4)      # the abstain counts against recall only
    assert list(s["wrong"]["reported_name"]) == ["C"]


def test_parse_check_catches_a_missing_row():
    funds = pd.DataFrame({"fund_id": ["F1"], "commitment_usd": [1.0], "called_usd": [1.0],
                          "nav_usd": [90.0]})
    ok = check_parse(TRUTH, funds, TRUTH, funds)
    assert all(r["status"] == "pass" for r in ok)
    bad = check_parse(TRUTH.iloc[:3], funds, TRUTH, funds)
    assert {r["check"] for r in bad if r["status"] == "fail"} == {
        "parse:F1 holdings count", "parse:F1 cost total", "parse:F1 fair value total"}


def test_nav_check():
    funds = pd.DataFrame({"fund_id": ["F1"], "nav_usd": [82.0]})
    assert check_nav(TRUTH, funds)[0]["status"] == "pass"      # 80 of 82 in holdings
    funds["nav_usd"] = [120.0]
    assert check_nav(TRUTH, funds)[0]["status"] == "fail"      # a third of NAV unexplained
    funds["nav_usd"] = [70.0]
    assert check_nav(TRUTH, funds)[0]["status"] == "fail"      # holdings worth more than NAV
