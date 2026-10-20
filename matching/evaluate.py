"""Score every matching method on the golden set, by query type and overall.

Smoke:  uv run python -m matching.evaluate --split dev --limit 20
Dev:    uv run python -m matching.evaluate --split dev       (also picks the hybrid thresholds)
Test:   uv run python -m matching.evaluate --split test      (run once, after dev)
LLM:    AWS_PROFILE=<profile> uv run python -m matching.evaluate --split dev --llm-rerank

Metrics (k = 5 candidates per query):
- top1     the first candidate is the right company
- top5     the right company is anywhere in the top 5 (recall@5)
- mrr      mean reciprocal rank: 1 for rank 1, 1/2 for rank 2 ... 0 if missing
- precision_answered, abstain_rate (hybrid only): how often it is right WHEN it answers,
  and how often it says "not sure"
- secs_per_1k_queries: wall time per 1,000 queries, on this laptop

Saves matching/results/eval_<split>.csv (summary) and matching/out/preds_<split>.csv
(one row per query and method). Logs to MLflow unless --limit is set.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from matching.common import (
    GOLDEN,
    OUT,
    RESULTS,
    THRESHOLDS,
    load_universe,
    start_mlflow,
)
from matching.matcher import METHODS, Matcher, decide

K = 5
SCORE_GRID = [70, 75, 80, 85, 88, 90, 92, 94, 96, 98, 100]
MARGIN_GRID = [0, 1, 2, 3, 5, 8, 10, 15]

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["dev", "test"], required=True,
                    help="tune on dev; score test once")
parser.add_argument("--limit", type=int, help="only N queries per type (smoke test)")
parser.add_argument("--target-precision", type=float, default=0.98,
                    help="dev picks the thresholds that answer most often at this precision")
parser.add_argument("--llm-rerank", action="store_true",
                    help="send low-margin hybrid queries to Claude Haiku on Bedrock")
parser.add_argument("--llm-budget", type=float, default=1.50, help="stop LLM calls at $X")
args = parser.parse_args()


def load_queries() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN)
    df = df[df["split"] == args.split]
    if args.limit:  # a few of every type, so the smoke test touches every code path
        df = df.groupby("type").sample(frac=1, random_state=14).groupby("type").head(args.limit)
    return df.reset_index(drop=True)


def rank_of(true_cik: int, cands: list[dict]) -> int | None:
    """1-based position of the right company, or None if it is not in the list."""
    return next((i + 1 for i, c in enumerate(cands) if c["cik"] == true_cik), None)


def pick_thresholds(preds: pd.DataFrame) -> dict:
    """Dev only: the (min_score, min_margin) pair that answers the most queries while
    staying at or above the target precision. A wrong link costs more than a 'not sure'."""
    best = None
    for s in SCORE_GRID:
        for m in MARGIN_GRID:
            answered = (preds["top1_score"] >= s) & (preds["margin"] >= m)
            if answered.sum() == 0:
                continue
            precision = preds.loc[answered, "correct1"].mean()
            row = {"min_score": s, "min_margin": m, "precision": round(float(precision), 4),
                   "coverage": round(float(answered.mean()), 4)}
            ok = precision >= args.target_precision
            key = (ok, row["coverage"] if ok else precision)
            if best is None or key > best[0]:
                best = (key, row)
    return best[1]


def summarize(preds: pd.DataFrame, method: str, secs: float) -> list[dict]:
    """One row per query type plus an 'all' row."""
    rows = []
    for qtype, g in [*preds.groupby("type"), ("all", preds)]:
        row = {"method": method, "type": qtype, "n": len(g), "top1": g["correct1"].mean(),
               "top5": g["rank"].notna().mean(), "mrr": (1 / g["rank"]).fillna(0).mean()}
        if "decision" in g:
            answered = g["decision"] == "match"
            row["abstain_rate"] = 1 - answered.mean()
            row["precision_answered"] = g.loc[answered, "correct1"].mean() if answered.any() \
                else float("nan")
        if qtype == "all":
            row["secs_per_1k_queries"] = secs / len(preds) * 1000
        rows.append(row)
    return rows


def to_preds(queries, ranked, method) -> pd.DataFrame:
    p = queries[["query", "cik", "type"]].copy()
    p["method"] = method
    p["rank"] = [rank_of(c, r) for c, r in zip(queries["cik"], ranked)]
    p["correct1"] = p["rank"] == 1
    p["top1_cik"] = [r[0]["cik"] if r else None for r in ranked]
    p["top1_name"] = [r[0]["name"] if r else None for r in ranked]
    p["top1_alias"] = [r[0]["alias"] if r else None for r in ranked]
    p["top1_score"] = [r[0]["score"] if r else 0.0 for r in ranked]
    p["margin"] = [r[0]["score"] - (r[1]["score"] if len(r) > 1 else 0) if r else 0.0
                   for r in ranked]
    return p


def llm_pass(queries, ranked, hybrid: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Re-ask the LLM on low-margin queries only; everything else keeps the hybrid answer."""
    from matching.llm_rerank import LLMReranker, needs_llm

    llm = LLMReranker(args.llm_budget)
    todo = [i for i, r in enumerate(ranked) if needs_llm(r)]
    print(f"\nLLM re-rank: {len(todo)} low-margin queries of {len(ranked)}")
    with ThreadPoolExecutor(4) as pool:  # network-bound calls, so overlap a few
        answers = list(pool.map(lambda i: llm.pick(queries["query"][i], ranked[i]), todo))
    p = hybrid.copy()
    p["method"] = "hybrid_llm"
    p["llm_status"] = ""
    for i, (cik, status) in zip(todo, answers):
        p.loc[i, "llm_status"] = status
        if status in ("error", "breaker", "budget"):
            continue  # no answer from the LLM: keep the hybrid's decision
        p.loc[i, "decision"] = "match" if cik else "abstain"
        p.loc[i, "top1_cik"] = cik
        p.loc[i, "top1_name"] = next((c["name"] for c in ranked[i] if c["cik"] == cik), None)
        p.loc[i, "correct1"] = cik == queries["cik"][i]
    # rank and top5 stay the hybrid's: the LLM only picks from those 5
    return p, llm


def main():
    queries = load_queries()
    matcher = Matcher(load_universe())
    print(f"{args.split}: {len(queries)} queries, universe {len(matcher.names)} companies, "
          f"{len(matcher.aliases)} aliases")

    all_preds, summary = [], []
    for method in METHODS:
        t0 = time.perf_counter()
        ranked = matcher.search_many(queries["query"].tolist(), method, K)
        secs = time.perf_counter() - t0
        p = to_preds(queries, ranked, method)
        if method == "hybrid":
            hybrid_ranked = ranked
            if args.split == "dev":
                t = pick_thresholds(p)
                print(f"hybrid thresholds chosen on dev: {t}")
                if not args.limit:
                    THRESHOLDS.write_text(json.dumps(t, indent=2))
            elif THRESHOLDS.exists():
                t = json.loads(THRESHOLDS.read_text())
            elif args.limit:  # smoke test before any dev run: matcher defaults
                t = {"min_score": matcher.min_score, "min_margin": matcher.min_margin}
            else:
                raise SystemExit("run --split dev first: it chooses the hybrid thresholds")
            p["decision"] = [decide(r, t["min_score"], t["min_margin"]) for r in ranked]
            # A wrong answer and an abstain both fail top1; precision_answered tells them apart.
            p["correct1"] = p["correct1"] & (p["decision"] == "match")
            hybrid_preds, thresholds = p, t
        all_preds.append(p)
        summary += summarize(p, method, secs)
        print(f"  {method:14s} {secs:6.1f}s")

    llm = None
    if args.llm_rerank:
        t0 = time.perf_counter()
        p, llm = llm_pass(queries, hybrid_ranked, hybrid_preds)
        all_preds.append(p)
        summary += summarize(p, "hybrid_llm", time.perf_counter() - t0)
        print(f"  LLM calls {llm.calls}, invalid answers {llm.invalid}, "
              f"cost ${llm.cost_usd:.4f}")

    table = pd.DataFrame(summary)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nTop-1 accuracy by type (hybrid: abstain counts as a miss)")
        print(table.pivot(index="type", columns="method", values="top1").round(3))
        print("\nOverall")
        print(table[table["type"] == "all"].drop(columns="type").round(3).to_string(index=False))

    preds = pd.concat(all_preds, ignore_index=True)
    stem = f"{args.split}" + ("_smoke" if args.limit else "")
    results_file = (OUT if args.limit else RESULTS) / f"eval_{stem}.csv"
    preds_file = OUT / f"preds_{stem}.csv"
    table.to_csv(results_file, index=False)
    preds.to_csv(preds_file, index=False)
    print(f"saved {results_file} and {preds_file}")

    if args.limit:
        print("smoke test: not logged to MLflow")
        return
    import mlflow

    start_mlflow()
    with mlflow.start_run(run_name=f"matching_{args.split}" + ("_llm" if llm else "")):
        mlflow.log_params({"split": args.split, "n_queries": len(queries), "k": K,
                           "universe_companies": len(matcher.names),
                           "aliases": len(matcher.aliases), "llm_rerank": args.llm_rerank,
                           "target_precision": args.target_precision, **thresholds})
        for r in summary:
            for metric in ["top1", "top5", "mrr", "abstain_rate", "precision_answered",
                           "secs_per_1k_queries"]:
                if metric in r and pd.notna(r[metric]):
                    mlflow.log_metric(f"{r['method']}.{r['type']}.{metric}", float(r[metric]))
        if llm:
            mlflow.log_metrics({"llm_calls": llm.calls, "llm_cost_usd": llm.cost_usd,
                                "llm_invalid": llm.invalid})
        mlflow.log_artifact(str(results_file))
        mlflow.log_artifact(str(preds_file))
    print("logged to MLflow experiment 'matching'")


if __name__ == "__main__":
    main()
