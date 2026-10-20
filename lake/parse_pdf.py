"""Read a fund report PDF back into rows: the fund summary, the holdings, the commentary.

Try one:  uv run python -m lake.parse_pdf data/lake/bronze/fund_reports/F01.pdf

This file only sees the PDF. It never opens the truth files: the lake has to be
built from documents alone, the way it would be with a real GP report.

How: pdfplumber finds the ruled tables on each page (it follows the grid lines),
and gives the page text for the commentary. We find each table by its header,
not by its position, so a report that moves a table to another page still parses.
"""

import re
import sys
from pathlib import Path

import pdfplumber

from lake.common import sector_code

# Summary labels -> our column names. Matched on the start of the label.
SUMMARY_FIELDS = {"fund name": "fund_name", "fund id": "fund_id", "vintage": "vintage",
                  "report date": "as_of", "total commitments": "commitment_usd",
                  "called capital": "called_usd", "net asset value": "nav_usd"}
HOLDING_HEADERS = {"portfolio company": "reported_name", "sector": "sector", "cost": "cost_usd",
                   "fair value": "fair_value_usd", "ownership": "ownership_pct"}


class ParseError(ValueError):
    """The PDF does not look like the report we expect: stop, don't guess."""


def clean(cell) -> str:
    """pdfplumber cells can be None or hold line breaks; make them one plain line."""
    return " ".join(str(cell or "").split())


def scale_of(label: str) -> float:
    """'Cost (USD 000s)' -> 1000; '(USD millions)' -> 1e6; no unit -> 1."""
    low = label.lower()
    if "000s" in low or "thousand" in low:
        return 1e3
    if "million" in low:
        return 1e6
    return 1.0


def to_number(text: str) -> float:
    """'1,234' -> 1234.0, '(56)' -> -56.0, '1.25%' -> 1.25. Raises on anything else."""
    s = clean(text).replace(",", "").replace("$", "").replace("%", "")
    negative = s.startswith("(") and s.endswith(")")
    value = float(s.strip("()"))  # ValueError if it is not a number
    return -value if negative else value


def parse_summary(table: list[list]) -> dict:
    out = {}
    for label, value in (row[:2] for row in table[1:]):
        label, value = clean(label), clean(value)
        field = next((f for key, f in SUMMARY_FIELDS.items() if label.lower().startswith(key)),
                     None)
        if field in ("commitment_usd", "called_usd", "nav_usd"):
            out[field] = to_number(value) * scale_of(label)
        elif field == "vintage":
            out[field] = int(value)
        elif field:
            out[field] = value
    missing = set(SUMMARY_FIELDS.values()) - set(out)
    if missing:
        raise ParseError(f"summary table is missing {sorted(missing)}")
    return out


def parse_holdings(table: list[list], page: int) -> tuple[list[dict], dict | None]:
    """Holdings rows plus the printed Total row (used to check we read every row)."""
    header = [clean(h) for h in table[0]]
    cols = {}
    for i, h in enumerate(header):
        field = next((f for key, f in HOLDING_HEADERS.items() if h.lower().startswith(key)), None)
        if field:
            cols[field] = (i, scale_of(h))
    if set(cols) != set(HOLDING_HEADERS.values()):
        raise ParseError(f"holdings header not recognized: {header}")
    rows, total = [], None
    for raw in table[1:]:
        cell = {f: clean(raw[i]) for f, (i, _) in cols.items()}
        if not cell["reported_name"]:
            continue
        money = {f: to_number(cell[f]) * cols[f][1] for f in ("cost_usd", "fair_value_usd")}
        if cell["reported_name"].lower() == "total":
            total = money
            continue
        rows.append({"reported_name": cell["reported_name"], "sector": sector_code(cell["sector"]),
                     **money, "ownership_pct": to_number(cell["ownership_pct"]),
                     "source_page": page})
    return rows, total


def parse_commentary(text: str) -> str:
    """The text after the 'Manager Commentary' heading, minus the page banner."""
    lines = (text or "").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Manager Commentary")
    except StopIteration:
        return ""
    body = [ln for ln in lines[start + 1:] if not ln.startswith("SYNTHETIC DOCUMENT")]
    return " ".join(" ".join(body).split())


def parse_fund_pdf(path) -> dict:
    """One PDF -> {"fund": {...}, "holdings": [...], "commentary": [{"page", "text"}]}.

    Raises ParseError when a part is missing or the rows don't add up to the
    printed Total: a half-read report must not reach the lake silently.
    """
    fund, holdings, totals, commentary = None, [], [], []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                first = clean(table[0][0]).lower() if table and table[0] else ""
                if first == "item":
                    fund = parse_summary(table)
                elif first.startswith("portfolio company"):
                    rows, total = parse_holdings(table, page.page_number)
                    holdings += rows
                    totals += [total] if total else []
            text = parse_commentary(page.extract_text())
            if text:
                commentary.append({"page": page.page_number, "text": text})
    if fund is None or not holdings:
        raise ParseError(f"{path}: no {'fund summary' if fund is None else 'holdings'} table")
    for field in ("cost_usd", "fair_value_usd"):
        rows_sum = sum(h[field] for h in holdings)
        printed = sum(t[field] for t in totals)
        if totals and abs(rows_sum - printed) > 0.5:
            raise ParseError(f"{path}: {field} rows add up to {rows_sum:,.0f}, "
                             f"but the report's Total says {printed:,.0f}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fund["as_of"]):
        raise ParseError(f"{path}: report date {fund['as_of']!r} is not YYYY-MM-DD")
    return {"fund": fund, "holdings": holdings, "commentary": commentary,
            "source_doc": f"bronze/fund_reports/{Path(path).name}"}


if __name__ == "__main__":
    parsed = parse_fund_pdf(sys.argv[1])
    print(parsed["fund"])
    for h in parsed["holdings"]:
        print(f"  p{h['source_page']}  {h['reported_name']:35s} {h['sector']:22s} "
              f"{h['cost_usd']:>14,.0f} {h['fair_value_usd']:>14,.0f} {h['ownership_pct']:5.2f}%")
    for c in parsed["commentary"]:
        print(f"\ncommentary (page {c['page']}): {c['text'][:300]}...")
