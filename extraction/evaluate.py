"""Score extraction against XBRL truth: (a) one LLM call vs (b) the LangGraph flow.

Smoke:  AWS_PROFILE=<your-profile> uv run python extraction/evaluate.py --variant baseline --limit 5
        AWS_PROFILE=<your-profile> uv run python extraction/evaluate.py --variant graph --limit 5
Run:    AWS_PROFILE=<your-profile> uv run python extraction/evaluate.py --variant baseline
        AWS_PROFILE=<your-profile> uv run python extraction/evaluate.py --variant graph

(a) baseline: the first 8,000 characters of Item 8, one call, no locate, no checks,
    no retry. It is what you would build in ten minutes.
(b) graph: graph.py, with locate, checks, one retry and a human-review route.

Both use the same model, prompt and tool, and the same scoring, so the difference
comes from the flow around the call. Per field we report:
  coverage      a value came back (out of filings where XBRL has the field)
  acc_1pct      within 1% of XBRL;  acc_5pct  within 5%
  scale_errors  off by about 1,000x or 1,000,000x (wrong "thousands"/"millions")
  sign_errors   right size, wrong sign (a loss reported as a profit)
  acc_1pct_not_flagged   accuracy on filings the flow did NOT send to review
plus the review rate, cost per 1,000 filings and latency.

Outputs: extraction/results/eval_<variant>.csv (summary), extraction/out/<variant>/
(one JSON per filing), extraction/out/preds_<variant>.csv, and for the graph
extraction/results/analyst_notes_sample.md. Smoke runs (--limit) write to
extraction/out/smoke/ and are not logged to MLflow.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm
import mlflow
import pandas as pd
from analysis import sector_reference
from common import FIELDS, OUT, RESULTS, SAMPLE_CSV, load_texts, start_mlflow
from graph import BUDGET, EXTRACT_PROMPT, build_graph, extract_call, normalize_fields

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["baseline", "graph"], required=True)
parser.add_argument("--limit", type=int, help="only the first N filings (smoke test)")
parser.add_argument("--workers", type=int, default=4, help="filings in flight at once")
args = parser.parse_args()

sample = pd.read_csv(SAMPLE_CSV)
if args.limit:
    sample = sample.head(args.limit)
keys = list(zip(sample["cik"], sample["year"]))
print(f"{args.variant}: {len(sample)} filings, model {llm.MODEL_ID}, {args.workers} at a time")
print("loading filing texts...")
texts = load_texts(set(keys))
ref = sector_reference() if args.variant == "graph" else {}
graph = build_graph()


def run_baseline(row, sections) -> dict:
    r = extract_call(sections["section_8"][:BUDGET])
    return {"status": "done", "reason": "baseline: no checks", "extracted": r["output"],
            "values": normalize_fields(r["output"]), "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"], "retries": 0, "trace": ["single_call"]}


def run_graph(row, sections) -> dict:
    prior = row["revenue_prior"]
    return graph.invoke({
        "cik": int(row["cik"]), "year": int(row["year"]), "name": row["name"],
        "sector": row["sector"], "sections": sections,
        "prior_revenue": None if pd.isna(prior) else float(prior),
        "sector_ref": ref.get(row["sector"], {}),
    })


def run_one(row) -> dict:
    """One filing -> its final state, plus wall-clock seconds. Never raises."""
    t0 = time.perf_counter()
    sections = texts.get((row["cik"], row["year"]), {})
    try:
        state = (run_baseline if args.variant == "baseline" else run_graph)(row, sections)
    except Exception as e:  # noqa: BLE001 (e.g. circuit breaker open): record it, keep going
        state = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:150]}"}
    state = {k: v for k, v in state.items() if k not in ("sections", "sector_ref")}
    return state | {"cik": int(row["cik"]), "year": int(row["year"]),
                    "seconds": time.perf_counter() - t0}


# ---------- run, with progress and ETA ----------

out_dir = OUT / "smoke" if args.limit else OUT
filing_dir = out_dir / args.variant
filing_dir.mkdir(parents=True, exist_ok=True)
results = {}
t0 = time.perf_counter()
with ThreadPoolExecutor(args.workers) as pool:  # LLM calls wait on the network, so overlap them
    futures = [pool.submit(run_one, row) for _, row in sample.iterrows()]
    for i, fut in enumerate(as_completed(futures), 1):
        s = fut.result()
        results[(s["cik"], s["year"])] = s
        (filing_dir / f"{s['cik']}_{s['year']}.json").write_text(json.dumps(s, indent=1, default=str))
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (len(futures) - i)
        print(f"  [{i}/{len(futures)}] {s['cik']} {s['year']} {s['status']:18s} "
              f"{s['seconds']:5.1f}s  elapsed {elapsed:.0f}s  eta {eta:.0f}s")
wall = time.perf_counter() - t0

# ---------- one row per filing ----------

rows = []
for (cik, year), row in zip(keys, sample.to_dict("records")):
    s = results[(cik, year)]
    values = s.get("values") or {}
    rows.append({
        "cik": cik, "year": year, "name": row["name"], "sector": row["sector"],
        "status": s["status"], "reason": s.get("reason"), "retries": s.get("retries", 0),
        "trace": " > ".join(s.get("trace", [])), "tokens_in": s.get("tokens_in", 0),
        "tokens_out": s.get("tokens_out", 0), "seconds": s["seconds"],
        **{f"pred_{f}": values.get(f) for f in FIELDS},
        **{f"true_{f}": row.get(f) for f in FIELDS},
        "note": s.get("note"),
    })
preds = pd.DataFrame(rows)
preds_file = out_dir / f"preds_{args.variant}.csv"
preds.to_csv(preds_file, index=False)


# ---------- scoring ----------

def near(ratio: pd.Series, k: float) -> pd.Series:
    return (ratio / k - 1).abs() < 0.05


def score_field(pred: pd.Series, true: pd.Series, ok: pd.Series) -> dict:
    """Compare one field. Only filings where XBRL has a non-zero value count."""
    has = true.notna() & (true != 0)
    p, t = pred[has].astype(float), true[has].astype(float)
    rel = (p - t).abs() / t.abs()      # NaN when no prediction -> counts as wrong
    ratio = (p / t).abs()
    within1 = rel <= 0.01
    return {
        "n_truth": int(has.sum()),
        "coverage": p.notna().mean(),
        "acc_1pct": within1.mean(),
        "acc_5pct": (rel <= 0.05).mean(),
        "scale_errors": int(sum(near(ratio, k) for k in (1e3, 1e6, 1e-3, 1e-6)).sum()),
        "sign_errors": int((((p + t).abs() / t.abs()) <= 0.01).sum()),
        "n_not_flagged": int(ok[has].sum()),
        "acc_1pct_not_flagged": within1[ok[has]].mean() if ok[has].any() else float("nan"),
    }


ok = preds["status"] == "done"
summary = pd.DataFrame([{"field": f, **score_field(preds[f"pred_{f}"], preds[f"true_{f}"], ok)}
                        for f in FIELDS])
n = len(preds)
cost = llm.cost(preds["tokens_in"].sum(), preds["tokens_out"].sum())
secs = sorted(preds["seconds"])
overall = {
    "n_filings": n,
    "review_rate": (preds["status"] == "needs_human_review").mean(),
    "error_rate": (preds["status"] == "error").mean(),
    "retry_rate": (preds["retries"] > 0).mean(),
    "cost_usd": cost,
    "cost_per_1k_usd": cost / n * 1000,
    "latency_median_s": statistics.median(secs),
    "latency_p95_s": float(pd.Series(secs).quantile(0.95)),
    "wall_seconds": wall,
    "avg_tokens_in": preds["tokens_in"].mean(),
    "avg_tokens_out": preds["tokens_out"].mean(),
}

pd.set_option("display.width", 160)
print("\n" + summary.round(3).to_string(index=False))
print()
for k, v in overall.items():
    print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")
if (preds["status"] == "needs_human_review").any():
    print("\nreview reasons (first 60 chars):")
    print(preds.loc[preds["status"] == "needs_human_review", "reason"].str[:60]
          .value_counts().head(10).to_string())

results_dir = out_dir if args.limit else RESULTS
results_dir.mkdir(parents=True, exist_ok=True)
eval_file = results_dir / f"eval_{args.variant}.csv"
pd.concat([summary, pd.DataFrame([{"field": "all", **overall}])]).to_csv(eval_file, index=False)


# ---------- analyst notes (graph only) ----------

def write_notes(path, k: int = 10) -> None:
    lines = ["# Analyst notes: sample", "",
             "Written by the LLM from computed numbers only; a code check confirms every",
             "number in a note is in the computed set. Outlier flags come from a simple",
             "statistical model (robust z-scores against sector medians of XBRL values).", ""]
    for _, r in preds[preds["note"].fillna("") != ""].head(k).iterrows():
        s = results[(r["cik"], r["year"])]
        outliers = [name for name, f in (s.get("flags") or {}).items() if f["outlier"]]
        lines += [f"## {r['name']} ({r['sector']}, {r['year']})", "",
                  f"Status: {r['status']} ({r['reason']})  ",
                  f"Outlier ratios: {', '.join(outliers) or 'none'}", "",
                  f"> {r['note']}", ""]
    path.write_text("\n".join(lines))


notes_file = results_dir / "analyst_notes_sample.md"
if args.variant == "graph":
    write_notes(notes_file)

# ---------- MLflow ----------

errors = int((preds["status"] == "error").sum())
if args.limit:
    print("\nsmoke test: not logged to MLflow")
elif errors > 0.05 * n:
    print(f"\n{errors} of {n} filings errored: not logged to MLflow (fix and rerun)")
else:
    start_mlflow()
    with mlflow.start_run(run_name=f"extraction_{args.variant}"):
        mlflow.log_params({"variant": args.variant, "model": llm.MODEL_ID, "n_filings": n,
                           "workers": args.workers, "chars_per_call": BUDGET,
                           "price_in_per_1m": llm.PRICE_IN, "price_out_per_1m": llm.PRICE_OUT})
        mlflow.log_metrics({k: float(v) for k, v in overall.items()})
        for r in summary.to_dict("records"):
            mlflow.log_metrics({f"{r['field']}_{k}": float(v) for k, v in r.items()
                                if k != "field" and pd.notna(v)})
        mlflow.log_text(EXTRACT_PROMPT, "extract_prompt.txt")
        for f in (eval_file, preds_file) + ((notes_file,) if args.variant == "graph" else ()):
            mlflow.log_artifact(str(f))
    print(f"\nlogged MLflow run extraction_{args.variant}")
print(json.dumps({"summary": str(eval_file), "per_filing": str(preds_file)}))
