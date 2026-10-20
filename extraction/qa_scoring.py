"""How the Q&A evaluation reads and grades numeric answers. Pure code, unit-tested.

Two rules worth knowing:
  - Revenue is graded in percent error; growth in percentage points ("12%" against a
    true 12.6% is a 0.6-point miss, not a 5% miss).
  - Abstaining ("not in the document") is correct when the number is not in the
    filing text, and wrong when it is.
"""

import re

MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand)?", re.IGNORECASE)
PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
DECREASE = re.compile(r"\b(decrease|declin|fell|lower|down|drop|reduc)", re.IGNORECASE)
SCALE = {"billion": 1e9, "million": 1e6, "thousand": 1e3}


def dollar_amounts(text: str) -> list[float]:
    """Every "$..." amount in a text, in dollars: "$7.94 billion" -> 7.94e9."""
    out = []
    for num, word in MONEY.findall(text):
        try:
            out.append(float(num.replace(",", "")) * SCALE.get(word.lower(), 1))
        except ValueError:
            continue
    return out


def in_text(amounts: list[float], truth: float) -> bool:
    """Is a dollar amount within 1% of the truth written anywhere in the text?"""
    return any(abs(a - truth) <= 0.01 * abs(truth) for a in amounts)


def predicted_value(kind: str, answer: str) -> float | None:
    """The number an answer gives: the first dollar amount (revenue) or the first
    percent (growth). "decreased 5%" becomes -5."""
    if kind == "revenue":
        amounts = dollar_amounts(answer)
        return amounts[0] if amounts else None
    m = PERCENT.search(answer)
    if not m:
        return None
    value = float(m.group(1))
    return -value if value > 0 and DECREASE.search(answer) else value


def grade(kind: str, pred: float | None, truth: float, abstained: bool, findable: bool,
          tol: float) -> tuple[bool, float | None]:
    """(correct?, error). Error is percent for revenue, percentage points for growth."""
    if abstained:
        return not findable, None
    if pred is None:
        return False, None
    err = abs(pred - truth) / abs(truth) * 100 if kind == "revenue" else abs(pred - truth)
    return err <= tol, err
