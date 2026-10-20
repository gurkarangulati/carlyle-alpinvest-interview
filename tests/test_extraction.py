"""Unit tests for the code-only parts of the extraction flow (no LLM, no network).

Run: uv run pytest tests/test_extraction.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))

from analysis import compute_ratios, flag_outliers, mad, robust_z, ungrounded_numbers
from checks import (
    apply_scale,
    evidence_found,
    number_in_quote,
    parse_number,
    plausibility_problems,
    year_problem,
)
from graph import find_problems, normalize_fields, ranked_chunks, route_after_check

# ---------- scale normalization ----------


@pytest.mark.parametrize(("written", "expected"), [
    ("1,234", 1234.0), ("(1,234)", -1234.0), ("$ (56.7)", -56.7), ("-89", -89.0),
    ("−12", -12.0), ("$2,000", 2000.0), ("—", None), ("n/a", None), (None, None),
])
def test_parse_number(written, expected):
    assert parse_number(written) == expected


@pytest.mark.parametrize(("written", "scale", "expected"), [
    ("1,234", "thousands", 1_234_000), ("(1,234)", "thousands", -1_234_000),
    ("2.5", "millions", 2_500_000), ("987", "units", 987),
    ("1.2 billion", "units", 1_200_000_000),  # a scale word in the value wins
    ("1,234", "lakhs", None),                 # unknown scale: no guess
])
def test_apply_scale(written, scale, expected):
    assert apply_scale(written, scale) == expected


# ---------- evidence ----------

TEXT = "CONSOLIDATED STATEMENTS OF OPERATIONS (in thousands)\nTotal revenues   $ 12,345   $ 11,000"


def test_evidence_exact_and_whitespace():
    assert evidence_found("Total revenues $ 12,345", TEXT)  # extra spaces squashed
    assert evidence_found("total REVENUES   $ 12,345", TEXT)  # case ignored


def test_evidence_made_up_quote_fails():
    assert not evidence_found("Net revenues increased 40% to a record", TEXT)
    assert not evidence_found("", TEXT)
    assert not evidence_found(None, TEXT)


def test_number_in_quote():
    assert number_in_quote("12,345", "Total revenues $ 12,345 $ 11,000")
    assert number_in_quote("(1,234)", "Net loss (1,234)")
    assert not number_in_quote("12,346", "Total revenues $ 12,345")
    assert not number_in_quote("1,234", "Total revenues $ 11,234")  # no partial matches


# ---------- plausibility and year ----------

def test_plausible_values_pass():
    assert plausibility_problems({"revenue": 5e8, "net_income": -2e7, "total_assets": 9e8}) == []


def test_assets_must_be_positive():
    assert plausibility_problems({"total_assets": 0})


def test_scale_slip_is_caught():
    # Net income read in units while revenue was in thousands: 1,000x too small revenue.
    assert plausibility_problems({"revenue": 5_000, "net_income": 4_000_000, "total_assets": 9e6})


def test_year_check():
    assert year_problem("2018-12-31", 2018) is None
    assert year_problem("2019-01-31", 2018) is None       # retailer year ending in January
    assert year_problem("2017-12-31", 2018)               # prior-year column
    assert year_problem("Dec 2018", 2018)                 # not a date


def test_find_problems_and_route():
    extracted = {"fiscal_year_end": "2018-12-31", "revenue_value": "12,345",
                 "revenue_scale": "thousands", "revenue_evidence": "Total revenues $ 12,345"}
    values = {"revenue": 12_345_000}
    assert find_problems(extracted, values, TEXT, 2018) == []
    extracted["revenue_evidence"] = "Revenues were 12,345"  # not in the text
    problems = find_problems(extracted, values, TEXT, 2018)
    assert any("evidence" in p for p in problems)
    assert route_after_check({"problems": problems, "retries": 0}) == "locate"
    assert route_after_check({"problems": problems, "retries": 1}) == "needs_human_review"
    assert route_after_check({"problems": [], "retries": 1}) == "analyze"


def test_normalize_fields_reads_flat_keys():
    values = normalize_fields({"net_income_value": "(4.2)", "net_income_scale": "millions"})
    assert values["net_income"] == -4_200_000
    assert values["revenue"] is None  # missing keys -> None, not a crash


def test_locate_ranks_statement_pages_first():
    sections = {"section_8": "Report of auditors. " * 200 + TEXT * 20, "section_7": "", "section_15": ""}
    best = ranked_chunks(sections)[0]
    assert best[1] == "section_8" and best[2] >= 2000  # the statements, not the audit letter


# ---------- robust z-scores ----------

def test_mad_and_robust_z():
    xs = [0.05, 0.06, 0.07, 0.08, 10.0]  # one wild value
    assert mad(xs) == pytest.approx(0.01)  # the wild value doesn't blow up the spread
    assert robust_z(0.07, 0.07, 0.01) == 0
    assert robust_z(0.12, 0.07, 0.01) == pytest.approx(0.6745 * 5)
    assert robust_z(0.12, 0.07, 0) is None  # no spread: no score


def test_ratios_and_outlier_flag():
    ratios = compute_ratios({"revenue": 100.0, "net_income": 50.0, "total_assets": 200.0}, 80.0)
    assert ratios == pytest.approx({"net_margin": 0.5, "return_on_assets": 0.25,
                                    "revenue_growth": 0.25})
    ref = {"net_margin": {"median": 0.05, "mad": 0.03, "n": 50}}
    flags = flag_outliers(ratios, ref)
    assert flags["net_margin"]["outlier"]          # z = 0.6745 * 0.45 / 0.03 = 10.1
    assert "return_on_assets" not in flags         # no reference: no flag


def test_ratios_skip_missing_inputs():
    assert compute_ratios({"net_income": 5.0}, None) == {
        "net_margin": None, "return_on_assets": None, "revenue_growth": None}


# ---------- note grounding ----------

FACTS = {"fiscal_year": "2018", "revenue": "$1.2 billion", "net_margin": "-3.4%",
         "sector_median_net_margin": "5.1%", "net_margin_robust_z": "-4.2",
         "outlier_rule": "|robust z| above 3.5"}


def test_grounded_note_passes():
    note = ("In fiscal 2018 the company earned $1.2 billion of revenue at a net margin of "
            "-3.4%, against a sector median of 5.1% (robust z -4.2, an outlier).")
    assert ungrounded_numbers(note, FACTS) == []


def test_invented_number_is_flagged():
    note = "Revenue was $1.2 billion, up 12% from last year."
    assert ungrounded_numbers(note, FACTS) == ["12"]


def test_rounded_number_is_flagged():
    assert ungrounded_numbers("Net margin was about -3%.", FACTS) == ["3"]
