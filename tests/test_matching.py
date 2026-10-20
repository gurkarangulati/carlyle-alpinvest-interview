"""Tests for the matching module. No network, no data files: a tiny made-up universe.

Run:  uv run python -m pytest tests/test_matching.py -q
"""

import random

import pytest

from matching.golden import make_variant, typo
from matching.matcher import Matcher, decide, normalize
from matching.mentions import find_mentions

# ---- normalize() ---------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("The Walt Disney Company", "walt disney"),
    ("AIR PRODUCTS & CHEMICALS INC /DE/", "air products and chemicals"),
    ("Air Products and Chemicals, Inc.", "air products and chemicals"),
    ("BROADWAY FINANCIAL CORP \\DE\\", "broadway financial"),
    ("BEL FUSE INC /NJ", "bel fuse"),
    ("Acme Intl Hldgs Corp.", "acme international holdings"),
    ("Acme International Holdings Corporation", "acme international holdings"),
    ("U.S. Steel Corp", "us steel"),
    ("Macy's, Inc.", "macys"),
    ("JPMorgan Chase & Co.", "jpmorgan chase"),
    ("Coca-Cola Co", "coca cola"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_keeps_a_name_that_is_only_a_suffix():
    assert normalize("Inc.") == "inc"  # never return an empty key


# ---- variant generator ---------------------------------------------------------


@pytest.mark.parametrize("kind, name, check", [
    ("the", "The Kroger Co.", lambda v: v == "Kroger Co."),
    ("the", "KROGER CO", lambda v: v == "The KROGER CO"),
    ("ampersand", "Procter & Gamble Co", lambda v: v == "Procter and Gamble Co"),
    ("ampersand", "Johnson and Johnson", lambda v: v == "Johnson & Johnson"),
    ("abbreviation", "Acme International Corp", lambda v: v == "Acme Intl Corp"),
    ("case_punct", "AAR CORP", lambda v: v == "Aar Corp"),
    ("case_punct", "Apple, Inc.", lambda v: v == "APPLE INC"),
    ("suffix", "Acme Widgets, Inc.",
     lambda v: v in {"Acme Widgets Incorporated", "Acme Widgets Corp", "Acme Widgets"}),
])
def test_variant_kinds(kind, name, check):
    v = make_variant(name, kind, random.Random(0))
    assert v is not None and check(v), v


def test_variant_returns_none_when_edit_does_not_apply():
    rng = random.Random(0)
    assert make_variant("Apple Inc", "ampersand", rng) is None
    assert make_variant("H&R Block Inc", "ampersand", rng) is None
    assert make_variant("Apple", "suffix", rng) is None


def test_typo_changes_exactly_one_word_and_keeps_its_first_letter():
    for seed in range(50):
        name = "Northern Widgets Corp"
        v = typo(name, random.Random(seed))
        changed = [(a, b) for a, b in zip(name.split(), v.split(), strict=True) if a != b]
        assert len(changed) == 1
        assert changed[0][0][0] == changed[0][1][0]
        assert abs(len(changed[0][0]) - len(changed[0][1])) <= 1


def test_variants_are_reproducible_with_a_seed():
    a = [make_variant("Acme Widgets Inc", "typo", random.Random(14)) for _ in range(3)]
    assert len(set(a)) == 1


# ---- abstain logic -------------------------------------------------------------


def cands(*scores):
    return [{"cik": i, "score": s} for i, s in enumerate(scores)]


def test_decide_matches_when_score_and_margin_are_high():
    assert decide(cands(97, 80), min_score=90, min_margin=3) == "match"


def test_decide_abstains_on_low_score():
    assert decide(cands(85, 40), min_score=90, min_margin=3) == "abstain"


def test_decide_abstains_on_small_margin():
    assert decide(cands(96, 95), min_score=90, min_margin=3) == "abstain"


def test_decide_handles_one_or_zero_candidates():
    assert decide(cands(95), min_score=90, min_margin=3) == "match"
    assert decide([], min_score=90, min_margin=3) == "abstain"


# ---- matcher on a tiny universe ------------------------------------------------

UNIVERSE = [
    {"cik": 1, "name": "TWDC Enterprises 18 Corp.", "tickers": [],
     "former_names": [{"name": "WALT DISNEY CO/", "from": "1996-01-01", "to": "2019-03-20"}]},
    {"cik": 2, "name": "Walt Disney Co", "tickers": ["DIS"], "former_names": []},
    {"cik": 3, "name": "Apple Inc.", "tickers": ["AAPL"], "former_names": [
        {"name": "APPLE COMPUTER INC", "from": "1994-01-01", "to": "2007-01-10"}]},
    {"cik": 4, "name": "Apple Hospitality REIT, Inc.", "tickers": ["APLE"], "former_names": []},
]


@pytest.fixture(scope="module")
def matcher():
    return Matcher(UNIVERSE, min_score=90, min_margin=3)


def test_former_name_finds_the_renamed_company(matcher):
    assert matcher.search("Apple Computer, Inc.", "hybrid")[0]["cik"] == 3
    assert matcher.search("Apple Computer, Inc.", "exact")[0]["cik"] == 3


def test_without_aliases_the_former_name_is_lost(matcher):
    top = matcher.search("Apple Computer Inc", "tfidf_noalias")
    assert all(c["alias_kind"] == "current" for c in top)


def test_same_name_twice_is_an_abstain(matcher):
    # "Walt Disney Co" is cik 2's current name AND cik 1's former name.
    assert matcher.match("Walt Disney Company")["decision"] == "abstain"


def test_as_of_uses_only_names_valid_on_that_date(matcher):
    # In 2015 "Walt Disney Co" was cik 1. Our toy cik 2 has no dates, so it counts as
    # always valid; the point is that after 2019 cik 1's old name is gone.
    late = matcher.search("Walt Disney Co", "exact", as_of="2020-06-30")
    assert [c["cik"] for c in late] == [2]
    early = matcher.search("Walt Disney Co", "exact", as_of="2015-06-30")
    assert {c["cik"] for c in early} == {1, 2}


def test_ticker_lookup_is_exact(matcher):
    assert matcher.match("AAPL")["candidates"][0]["cik"] == 3
    assert matcher.match("AAPL")["decision"] == "match"


def test_typo_still_matches(matcher):
    assert matcher.match("Apple Hospitaltiy REIT")["candidates"][0]["cik"] == 4


# ---- mention finder ------------------------------------------------------------


def test_find_mentions():
    text = ("BUSINESS General North Texas Energy, Inc. (the Company) competes with "
            "Procter & Gamble Co. and Bank of America Corporation. We sell to stores.")
    found = [m["mention"] for m in find_mentions(text)]
    assert found == ["North Texas Energy, Inc.", "Procter & Gamble Co.",
                     "Bank of America Corporation"]
