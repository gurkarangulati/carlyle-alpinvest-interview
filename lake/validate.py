"""Data-quality checks on the built lake. Prints a report, writes lake/results/validate.csv.

Run:  uv run python -m lake.validate

Two kinds of check:
  hard  the lake is wrong or unusable if this fails (missing keys, broken schema,
        a PDF read incompletely, wrong matches). The script exits 1.
  soft  worth knowing, not a reason to stop (the LLM disagreeing with XBRL on some
        numbers, a matcher that abstains a little too often). Reported as "warn".

This is the one lake script that reads the truth files: it grades the lake
against them. The lake itself never sees them.
"""

import sys

import duckdb
import pandas as pd

from lake.common import DUCKDB_FILE, GOLD_VIEWS, GOLDEN, KEYS, RESULTS, SILVER_TABLES

MIN_PRECISION = 0.95       # a wrong match puts a company in the wrong fund: hard
MIN_RECALL = 0.90          # an abstain is visible and fixable by a person: soft
MAX_CASH_SHARE = 0.10      # NAV above the holdings' fair value = cash etc.; more is suspect
MIN_LLM_AGREEMENT = 0.80   # share of passed extractions within 1% of XBRL: soft


def result(check: str, level: str, ok: bool, value, threshold="", detail="") -> dict:
    status = "pass" if ok else ("fail" if level == "hard" else "warn")
    return {"check": check, "level": level, "status": status, "value": value,
            "threshold": threshold, "detail": detail}


# ---- structure -----------------------------------------------------------------

def check_schema(con) -> list[dict]:
    """Every contract table exists with exactly the agreed columns and types."""
    out = []
    for table, cols in SILVER_TABLES.items():
        actual = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table}").fetchall()]
        out.append(result(f"schema:{table}", "hard", actual == cols, len(actual), len(cols),
                          "" if actual == cols else f"got {actual}"))
    return out


def check_row_counts(con) -> list[dict]:
    out = []
    for t in list(SILVER_TABLES) + GOLD_VIEWS:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        level = "soft" if t == "metric_reconciliation" else "hard"  # empty if built --no-llm
        out.append(result(f"rows:{t}", level, n > 0, n, "> 0"))
    return out


def check_keys(con) -> list[dict]:
    out = []
    for table, cols in KEYS.items():
        nulls = con.execute(f"SELECT count(*) FROM {table} WHERE "
                            + " OR ".join(f"{c} IS NULL" for c in cols)).fetchone()[0]
        out.append(result(f"null_keys:{table}", "hard", nulls == 0, nulls, 0, ",".join(cols)))
    unique = {"companies": "cik", "funds": "fund_id", "doc_chunks": "chunk_id",
              "metrics": "cik, fiscal_year, metric, source"}
    for table, key in unique.items():
        dupes = con.execute(f"SELECT count(*) - count(DISTINCT ({key})) FROM {table}").fetchone()[0]
        out.append(result(f"unique:{table}({key})", "hard", dupes == 0, dupes, 0))
    orphans = con.execute("SELECT count(*) FROM holdings WHERE cik IS NOT NULL "
                          "AND cik NOT IN (SELECT cik FROM companies)").fetchone()[0]
    out.append(result("holdings.cik in companies", "hard", orphans == 0, orphans, 0,
                      "every matched holding must exist in the entity master"))
    return out


# ---- the documents were read completely ----------------------------------------

def check_parse(holdings: pd.DataFrame, funds: pd.DataFrame, truth: pd.DataFrame,
                funds_truth: pd.DataFrame) -> list[dict]:
    """Per fund: same number of holdings and the same cost / fair-value totals as the truth."""
    out = []
    lake = holdings.groupby("fund_id").agg(n=("reported_name", "size"), cost=("cost_usd", "sum"),
                                           fv=("fair_value_usd", "sum"))
    true = truth.groupby("fund_id").agg(n=("reported_name", "size"), cost=("cost_usd", "sum"),
                                        fv=("fair_value_usd", "sum"))
    for fid in sorted(true.index):
        got = lake.loc[fid] if fid in lake.index else pd.Series({"n": 0, "cost": 0, "fv": 0})
        for col, label in (("n", "holdings count"), ("cost", "cost total"),
                           ("fv", "fair value total")):
            ok = abs(float(got[col]) - float(true.loc[fid, col])) < 1
            out.append(result(f"parse:{fid} {label}", "hard", ok, float(got[col]),
                              float(true.loc[fid, col])))
    merged = funds_truth.merge(funds, on="fund_id", how="left", suffixes=("_true", ""))
    for col in ("commitment_usd", "called_usd", "nav_usd"):
        bad = merged[(merged[col] - merged[f"{col}_true"]).abs().fillna(1) >= 1]["fund_id"]
        out.append(result(f"parse:funds {col}", "hard", bad.empty, len(bad), 0,
                          ",".join(bad)))
    return out


# ---- matching: precision and recall against the truth --------------------------

def matching_scores(holdings: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """precision = right matches / matches made; recall = right matches / all holdings.

    An abstain lowers recall but not precision: saying "I don't know" is not wrong.
    """
    m = truth.merge(holdings[["fund_id", "reported_name", "cik", "match_decision"]],
                    on=["fund_id", "reported_name"], how="left", suffixes=("_true", ""))
    made = m["match_decision"] == "match"
    right = made & (m["cik"] == m["cik_true"])
    by_kind = (m.assign(right=right).groupby("name_kind")["right"].agg(["sum", "size"])
               if "name_kind" in m else pd.DataFrame())
    return {"n": len(m), "matched": int(made.sum()), "right": int(right.sum()),
            "precision": right.sum() / made.sum() if made.sum() else 0.0,
            "recall": right.sum() / len(m) if len(m) else 0.0,
            "wrong": m[made & ~right][["fund_id", "reported_name", "cik", "cik_true"]],
            "by_kind": by_kind}


def check_matching(holdings: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    s = matching_scores(holdings, truth)
    wrong = "; ".join(f"{r.reported_name}->{r.cik}" for r in s["wrong"].itertuples())
    return [result("matching precision", "hard", s["precision"] >= MIN_PRECISION,
                   round(s["precision"], 4), MIN_PRECISION,
                   f"{s['right']}/{s['matched']} matches right. {wrong}".strip()),
            result("matching recall", "soft", s["recall"] >= MIN_RECALL, round(s["recall"], 4),
                   MIN_RECALL, f"{s['right']}/{s['n']} holdings resolved correctly")]


# ---- numbers -------------------------------------------------------------------

def check_extraction(con) -> list[dict]:
    """How often the LLM's number agrees with XBRL within 1%, per metric."""
    df = con.execute("""
        SELECT r.metric, e.status, r.agrees_within_1pct
        FROM metric_reconciliation r
        JOIN metrics e ON e.cik = r.cik AND e.fiscal_year = r.fiscal_year
                      AND e.metric = r.metric AND e.source = 'extracted'
        WHERE r.xbrl_value IS NOT NULL""").df()
    out = []
    for metric in ("revenue", "net_income", "total_assets"):
        g = df[df["metric"] == metric]
        passed = g[g["status"] == "ok"]
        rate = passed["agrees_within_1pct"].fillna(False).mean() if len(passed) else 0.0
        out.append(result(f"extracted vs xbrl:{metric}", "soft", rate >= MIN_LLM_AGREEMENT,
                          round(float(rate), 3), MIN_LLM_AGREEMENT,
                          f"{int(passed['agrees_within_1pct'].fillna(False).sum())}/{len(passed)} "
                          f"status=ok rows within 1%; {int((g['status'] == 'needs_human_review').sum())} "
                          f"sent to review, {int((g['status'] == 'missing').sum())} missing"))
    return out


def check_nav(holdings: pd.DataFrame, funds: pd.DataFrame) -> list[dict]:
    """NAV = holdings at fair value + cash and other net assets, so NAV >= sum(FV),
    and the difference should be small."""
    fv = holdings.groupby("fund_id")["fair_value_usd"].sum()
    out = []
    for f in funds.itertuples():
        gap = (f.nav_usd - fv.get(f.fund_id, 0.0)) / f.nav_usd
        out.append(result(f"nav vs fair value:{f.fund_id}", "hard", 0 <= gap <= MAX_CASH_SHARE,
                          round(gap, 4), f"0..{MAX_CASH_SHARE}", "share of NAV not in holdings"))
    return out


def run_checks(con, truth: pd.DataFrame, funds_truth: pd.DataFrame) -> list[dict]:
    holdings = con.execute("SELECT * FROM holdings").df()
    funds = con.execute("SELECT * FROM funds").df()
    return (check_schema(con) + check_row_counts(con) + check_keys(con)
            + check_parse(holdings, funds, truth, funds_truth)
            + check_matching(holdings, truth) + check_extraction(con)
            + check_nav(holdings, funds))


def main():
    con = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    truth = pd.read_csv(GOLDEN / "fund_truth.csv")
    funds_truth = pd.read_csv(GOLDEN / "funds_truth.csv")
    results = pd.DataFrame(run_checks(con, truth, funds_truth))
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "validate.csv", index=False)

    shown = results.drop(columns="detail").assign(value=results["value"].map(lambda v: f"{v:,.4g}"))
    print(shown.to_string(index=False))
    s = matching_scores(con.execute("SELECT * FROM holdings").df(), truth)
    print(f"\nmatching on {s['n']} holdings: precision {s['precision']:.3f}, "
          f"recall {s['recall']:.3f}  (correct, by reported-name kind:)")
    print(s["by_kind"].rename(columns={"sum": "correct", "size": "holdings"}).to_string())
    for r in results[results["status"] != "pass"].itertuples():
        print(f"{r.status.upper():5s} {r.check}: {r.value} (want {r.threshold}) {r.detail}")
    hard_fails = (results["status"] == "fail").sum()
    print(f"\n{(results['status'] == 'pass').sum()} pass, {(results['status'] == 'warn').sum()} "
          f"warn, {hard_fails} hard fail -> {RESULTS.name}/validate.csv")
    sys.exit(1 if hard_fails else 0)


if __name__ == "__main__":
    main()
