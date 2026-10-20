"""The SYNTHETIC fund universe: 5 invented funds holding ~50 real public companies.

Run:  uv run python -m lake.funds

What it writes
  data/lake/bronze/fund_reports/{fund_id}.pdf   one quarterly report per fund (Q4 2019)
  data/lake/golden/fund_truth.csv               every holding with its TRUE cik
  data/lake/golden/funds_truth.csv              every fund's true summary numbers

Why synthetic: real GP reports are confidential. So the funds, their values and
the PDFs are invented (and every page says so), but the companies they hold are
real SEC filers with real 10-Ks and XBRL numbers. That makes the hard part real:
the report names a company the way a GP would ("Facebook Inc", "Rexnord Corp",
a typo), and the lake has to work out which SEC company that is.

The truth files are the answer key. The lake (parse_pdf.py, build.py) never reads
them; only validate.py and golden_questions.py do.
"""

import json
import random

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from lake.common import AS_OF, DATA, FUND_REPORTS, GOLDEN, XBRL_CACHE, sector_label, xbrl_values
from matching.golden import clean_display, make_variant
from matching.matcher import normalize

SEED = 14
QUARTER = "Q4 2019"
BANNER = "SYNTHETIC DOCUMENT - fictional fund for training; values are invented"

# Five invented funds. Each lists the CIKs it holds (real companies, chosen because
# each has a fiscal-2019 10-K in data/raw/edgar and XBRL revenue + net income).
# Four companies are held by two funds: 723125 Micron, 277948 CSX,
# 1091667 Charter, 314808 Valaris.
FUNDS = [
    {"fund_id": "F01", "fund_name": "Larkspur Hollow Growth Fund II", "vintage": 2016,
     "ciks": [1326801, 723125, 849399, 1327567, 1352010, 877890, 780571, 1800, 899866,
              927066, 1610950]},
    {"fund_id": "F02", "fund_name": "Cobaltine Industrial Partners III", "vintage": 2015,
     "ciks": [1175454, 1439288, 1018963, 32604, 277948, 1494319, 1307954, 1164727, 1163302,
              708819, 79879]},
    {"fund_id": "F03", "fund_name": "Bramblecrest Consumer & Media Fund I", "vintage": 2014,
     "ciks": [56873, 935703, 866787, 1335258, 1116132, 1091667, 29989, 908937, 43196,
              1490281, 64803]},
    {"fund_id": "F04", "fund_name": "Tidewrack Energy & Real Assets Fund", "vintage": 2013,
     "ciks": [314808, 1451505, 1458891, 81362, 1552000, 1318220, 822818, 1398987, 765880,
              1289490, 912593]},
    {"fund_id": "F05", "fund_name": "Quillfeather Diversified Opportunities Fund IV",
     "vintage": 2018,
     "ciks": [1156039, 19617, 1230245, 354190, 91576, 1126328, 723125, 277948, 1091667,
              314808]},
]

# How a GP might write a company's name, with how often we use each form.
# "former_*" is used instead whenever a former name was still valid on the
# report date (the company renamed itself AFTER 2019-12-31).
NAME_KINDS = {"current": 0.35, "suffix": 0.25, "case_punct": 0.15, "abbreviation": 0.05,
              "typo": 0.20}


# ---- inputs --------------------------------------------------------------------

def load_companies(ciks: list[int]) -> dict[int, dict]:
    """SEC name history + sector + XBRL facts for each CIK we hold."""
    names = {}
    for line in (DATA / "sec_names.jsonl").read_text().splitlines():
        r = json.loads(line)
        names[r["cik"]] = r
    meta = pd.read_csv(DATA / "companies.csv", usecols=["cik", "year", "sector", "sic"])
    meta = meta[meta["year"] == 2019].set_index("cik")
    out = {}
    for cik in ciks:
        facts = json.loads((XBRL_CACHE / f"{cik}.json").read_text())
        out[cik] = {"cik": cik, "name": names[cik]["name"],
                    "former_names": names[cik]["formerNames"],
                    "sector": meta.loc[cik, "sector"], "x2019": xbrl_values(facts, 2019)}
    return out


# ---- reported names ------------------------------------------------------------

def former_name_on(company: dict, as_of: str) -> str | None:
    """A former name that was the company's legal name on `as_of`, if any.

    Example: Meta Platforms filed as "Facebook Inc" until 2021-10-27, so a
    2019-12-31 report would say Facebook. Former names that only differ by
    case or tags (/DE/) are not real renames and are skipped.
    """
    current = normalize(company["name"])
    for f in company["former_names"]:
        start, end = (f.get("from") or "")[:10], (f.get("to") or "")[:10]
        if (start or "0000") <= as_of <= (end or "9999") and normalize(f["name"]) != current:
            return clean_display(f["name"])
    return None


def reported_name(company: dict, rng: random.Random, as_of: str = AS_OF) -> tuple[str, str]:
    """The name as an LP would read it in the report, and which kind of name it is.

    Returns (name, kind). kind is one of: former_name, former_variant, current,
    suffix, case_punct, abbreviation, typo. The variant edits come from the
    matching module's golden-set generator, so both use the same realistic edits.
    """
    former = former_name_on(company, as_of)
    if former:
        if rng.random() < 0.3 and (v := make_variant(former, "suffix", rng)):
            return v, "former_variant"
        return former, "former_name"
    current = clean_display(company["name"])
    kind = rng.choices(list(NAME_KINDS), weights=list(NAME_KINDS.values()))[0]
    v = make_variant(current, kind, rng) if kind != "current" else None
    if v and not v.rstrip(" ,").endswith("&"):  # "Arthur J. Gallagher & Co." minus "Co." reads wrong
        return v, kind
    return current, "current"


# ---- values --------------------------------------------------------------------

def make_fund(fund: dict, companies: dict[int, dict], rng: random.Random) -> tuple[dict, list]:
    """Invent the numbers for one fund. Everything is a whole number of US$ thousands,
    because the report prints thousands and the round trip should be exact."""
    holdings = []
    for cik in fund["ciks"]:
        c = companies[cik]
        name, kind = reported_name(c, rng)
        cost_k = rng.randrange(15_000, 120_000)
        fv_k = round(cost_k * rng.uniform(0.55, 2.4))      # gain or loss vs. cost
        holdings.append({"fund_id": fund["fund_id"], "reported_name": name, "name_kind": kind,
                         "cik": cik, "true_name": c["name"], "sector": c["sector"],
                         "cost_usd": cost_k * 1000.0, "fair_value_usd": fv_k * 1000.0,
                         "ownership_pct": round(rng.uniform(0.05, 3.0), 2), "as_of": AS_OF})
    cost = sum(h["cost_usd"] for h in holdings) / 1000
    fv = sum(h["fair_value_usd"] for h in holdings) / 1000
    called_k = round(cost * rng.uniform(1.03, 1.10))        # fees and expenses on top of cost
    commitment_k = round(called_k / rng.uniform(0.82, 0.95) / 25_000) * 25_000
    nav_k = round(fv * (1 + rng.uniform(0.01, 0.04)))       # plus a little cash
    summary = {"fund_id": fund["fund_id"], "fund_name": fund["fund_name"],
               "vintage": fund["vintage"], "commitment_usd": commitment_k * 1000.0,
               "called_usd": called_k * 1000.0, "nav_usd": nav_k * 1000.0, "as_of": AS_OF,
               "n_holdings": len(holdings)}
    return summary, holdings


# ---- commentary: facts from real data only ------------------------------------

def money(x: float) -> str:
    return f"${x / 1e9:.1f} billion" if abs(x) >= 1e9 else f"${x / 1e6:.0f} million"


def rename_fact(company: dict, as_of: str) -> str | None:
    """'X filed with the SEC as Y until Month YYYY': only for renames before `as_of`."""
    current = normalize(company["name"])
    done = [f for f in company["former_names"]
            if f.get("to") and "2019-01-01" <= f["to"][:10] <= as_of
            and normalize(f["name"]) != current]
    if not done:
        return None
    f = max(done, key=lambda f: f["to"])
    month = pd.Timestamp(f["to"][:10]).strftime("%B %Y")
    return f"filed with the SEC as {clean_display(f['name'])} until {month}"


def commentary(holdings: list[dict], companies: dict[int, dict]) -> list[str]:
    """Two or three short paragraphs naming holdings, built only from XBRL and SEC names."""
    named = [h for h in holdings if h["name_kind"] != "typo"]
    by_fv = sorted(named, key=lambda h: -h["fair_value_usd"])
    growth, losses, renames = [], [], []
    for h in by_fv:
        x = companies[h["cik"]]["x2019"]
        if x.get("revenue") and x.get("revenue_prior"):
            g = 100 * (x["revenue"] / x["revenue_prior"] - 1)
            growth.append(f"{h['reported_name']} reported fiscal 2019 revenue of "
                          f"{money(x['revenue'])}, {'up' if g >= 0 else 'down'} "
                          f"{abs(g):.1f}% from the prior year")
        if (x.get("net_income") or 0) < 0:
            losses.append(f"{h['reported_name']} ({money(-x['net_income'])})")
        if (fact := rename_fact(companies[h["cik"]], AS_OF)):
            renames.append(f"{h['reported_name']} {fact}")
    paras = ["Portfolio performance. " + "; ".join(growth[:3]) + ". Figures are taken from "
             "each company's annual report on Form 10-K."]
    if losses:
        paras.append("Companies reporting a net loss for fiscal 2019: " + ", ".join(losses[:4])
                     + ". The manager continues to monitor these positions.")
    if renames:
        paras.append("Corporate actions. " + "; ".join(renames) + ".")
    return paras


# ---- the PDF -------------------------------------------------------------------

def _banner(canvas, doc):
    """Drawn on every page: the SYNTHETIC warning at the top and the bottom."""
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.red)
    canvas.drawCentredString(letter[0] / 2, letter[1] - 0.5 * inch, BANNER)
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"{BANNER} - page {doc.page}")
    canvas.restoreState()


GRID = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                   ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                   ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                   ("ALIGN", (2, 1), (-1, -1), "RIGHT")])  # numbers right, headers left


def write_fund_pdf(summary: dict, holdings: list[dict], paras: list[str], path) -> None:
    """Page 1 cover, page 2 summary + commentary, page 3 schedule of investments.

    All amounts are printed in US$ thousands, as most GP reports do; the
    parser must notice "(USD 000s)" and multiply back.
    """
    styles = getSampleStyleSheet()
    big = ParagraphStyle("big", parent=styles["Title"], fontSize=24, leading=30)
    warn = ParagraphStyle("warn", parent=styles["Title"], textColor=colors.red, fontSize=16)
    k = lambda usd: f"{usd / 1000:,.0f}"  # noqa: E731  (dollars -> "12,345" thousands)

    story = [Spacer(1, 2 * inch), Paragraph(summary["fund_name"], big),
             Paragraph(f"Quarterly Report to Limited Partners - {QUARTER}", styles["Heading2"]),
             Paragraph("As of December 31, 2019", styles["Heading3"]), Spacer(1, inch),
             Paragraph("SYNTHETIC - NOT A REAL FUND", warn),
             Paragraph("This fund, its general partner and every value in this report are "
                       "invented for a training exercise. The portfolio companies are real "
                       "public companies; their financial figures come from public filings.",
                       styles["Normal"]),
             PageBreak()]

    rows = [["Item", "Value"], ["Fund name", summary["fund_name"]],
            ["Fund ID", summary["fund_id"]], ["Vintage year", str(summary["vintage"])],
            ["Report date", summary["as_of"]],
            ["Total commitments (USD 000s)", k(summary["commitment_usd"])],
            ["Called capital (USD 000s)", k(summary["called_usd"])],
            ["Net asset value (USD 000s)", k(summary["nav_usd"])]]
    story += [Paragraph("Fund Summary", styles["Heading1"]),
              Table(rows, colWidths=[200, 260], style=GRID), Spacer(1, 0.3 * inch),
              Paragraph("Manager Commentary", styles["Heading1"])]
    story += [Paragraph(p, styles["Normal"]) for p in paras]
    story.append(PageBreak())

    rows = [["Portfolio company (as reported)", "Sector", "Cost (USD 000s)",
             "Fair value (USD 000s)", "Ownership"]]
    for h in holdings:
        rows.append([h["reported_name"], sector_label(h["sector"]), k(h["cost_usd"]),
                     k(h["fair_value_usd"]), f"{h['ownership_pct']:.2f}%"])
    rows.append(["Total", "", k(sum(h["cost_usd"] for h in holdings)),
                 k(sum(h["fair_value_usd"] for h in holdings)), ""])
    story += [Paragraph("Schedule of Investments", styles["Heading1"]),
              Table(rows, colWidths=[180, 100, 85, 105, 60], style=GRID, repeatRows=1)]

    # invariant=1: no creation timestamp, so the same inputs give the same bytes.
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"{summary['fund_name']} "
                            f"{QUARTER} (SYNTHETIC)", invariant=1)
    doc.build(story, onFirstPage=_banner, onLaterPages=_banner)


def main():
    rng = random.Random(SEED)
    companies = load_companies(sorted({c for f in FUNDS for c in f["ciks"]}))
    FUND_REPORTS.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    summaries, all_holdings = [], []
    for fund in FUNDS:
        summary, holdings = make_fund(fund, companies, rng)
        write_fund_pdf(summary, holdings, commentary(holdings, companies),
                       FUND_REPORTS / f"{fund['fund_id']}.pdf")
        summaries.append(summary)
        all_holdings += holdings
    pd.DataFrame(summaries).to_csv(GOLDEN / "funds_truth.csv", index=False)
    truth = pd.DataFrame(all_holdings)
    truth.to_csv(GOLDEN / "fund_truth.csv", index=False)

    print(f"{len(FUNDS)} SYNTHETIC fund reports -> {FUND_REPORTS.relative_to(DATA.parent)}")
    print(f"{len(truth)} holdings, {truth['cik'].nunique()} distinct companies, "
          f"{(truth.groupby('cik').size() > 1).sum()} held by two funds")
    print("reported-name kinds:\n" + truth["name_kind"].value_counts().to_string())
    print("sectors:\n" + truth.drop_duplicates("cik")["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
