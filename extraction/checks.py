"""Plain-code checks on what the LLM returned. No LLM calls here, so all of it is unit-tested.

Why check in code: Bedrock does not strictly enforce a tool's JSON schema, and even
a well-formed answer can be wrong (prior-year column, wrong scale, a made-up quote).
Each check below catches one of those mistakes.
"""

import re
import unicodedata
from datetime import date

from rapidfuzz import fuzz

SCALES = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
MAX_RATIO = 100  # loose on purpose: catches 1,000x scale slips, not odd businesses


# ---------- normalize: text -> number ----------

def parse_number(written) -> float | None:
    """'(1,234.5)' -> -1234.5, '$ 2,000' -> 2000.0, '—' -> None.

    Accounting tables show negatives in parentheses, so "(" ... ")" means minus.
    """
    if written is None:
        return None
    s = str(written).strip()
    negative = ("(" in s and ")" in s) or s.lstrip("$ ").startswith(("-", "−", "–"))
    digits = re.sub(r"[^\d.]", "", s)
    try:
        value = float(digits)
    except ValueError:
        return None  # a dash, "n/a", or empty: no number
    return -value if negative else value


def apply_scale(written, scale: str | None) -> float | None:
    """Turn a number as written plus its scale into dollars.

    A scale word inside the value ("1.2 billion") wins over the scale field,
    because it is the more local, more specific statement.
    """
    value = parse_number(written)
    if value is None:
        return None
    for word, mult in SCALE_WORDS.items():
        if word in str(written).lower():
            return value * mult
    mult = SCALES.get(str(scale).lower().strip())
    return None if mult is None else value * mult


# ---------- check 1: evidence ----------

def _squash(s: str) -> str:
    """Lowercase, unify quotes/dashes/spaces, so formatting noise doesn't fail a quote."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate(str.maketrans("’‘“”–—−", "''\"\"---"))
    return re.sub(r"\s+", " ", s).strip()


def evidence_found(quote: str | None, text: str) -> bool:
    """Is the model's quote really in the text it was shown?

    Exact match after squashing; else a fuzzy match of 95+ (allows a dropped
    character or two from the flattened tables, not a made-up sentence).
    """
    if not quote or not quote.strip():
        return False
    q, t = _squash(quote), _squash(text)
    return q in t or fuzz.partial_ratio(q, t) >= 95


def number_in_quote(written, quote: str | None) -> bool:
    """Does the quote contain the number itself? Stops 'right quote, wrong number'."""
    core = re.sub(r"[^\d.]", "", str(written or "")).strip(".")
    numbers = {re.sub(r"[^\d.]", "", n).strip(".") for n in re.findall(r"\d[\d,.]*", quote or "")}
    return bool(core) and core in numbers


# ---------- check 2: plausibility ----------

def plausibility_problems(v: dict) -> list[str]:
    """Simple accounting sanity checks on values in dollars."""
    rev, ni, assets = v.get("revenue"), v.get("net_income"), v.get("total_assets")
    problems = []
    if assets is not None and assets <= 0:
        problems.append("total assets must be above zero")
    if rev is not None and rev < 0:
        problems.append("revenue is negative")
    if rev and ni is not None and abs(ni) > MAX_RATIO * abs(rev):
        problems.append(f"|net income| is over {MAX_RATIO}x revenue (scale slip?)")
    if assets and assets > 0 and rev and rev > MAX_RATIO * assets:
        problems.append(f"revenue is over {MAX_RATIO}x total assets (scale slip?)")
    if assets and assets > 0 and ni is not None and abs(ni) > MAX_RATIO * assets:
        problems.append(f"|net income| is over {MAX_RATIO}x total assets (scale slip?)")
    return problems


# ---------- check 3: year ----------

def year_problem(fiscal_year_end: str | None, year: int) -> str | None:
    """The fiscal year end should belong to the filing's year.

    In this corpus a 10-K for `year` ends between Jan 1 of `year` and mid
    `year`+1 (retailers often end in late January). A date a year earlier
    usually means the model read the prior-year column.
    """
    try:
        fye = date.fromisoformat(str(fiscal_year_end)[:10])
    except ValueError:
        return f"fiscal year end {fiscal_year_end!r} is not a YYYY-MM-DD date"
    if not date(year, 1, 1) <= fye <= date(year + 1, 6, 30):
        return f"fiscal year end {fye} does not fit filing year {year}"
    return None
