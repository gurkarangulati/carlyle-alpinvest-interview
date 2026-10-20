"""Score the analyst agent on the golden questions, locally or against the deployed runtime.

    uv run python agent/evaluate.py --target local              # all questions, logs to MLflow
    uv run python agent/evaluate.py --target local --limit 5    # smoke run, no MLflow
    uv run python agent/evaluate.py --target aws                # through invoke.py

Golden file: data/lake/golden/questions.jsonl, one JSON per line:
    {id, type, question, expected, answer_kind, tolerance, expected_sources}

Scoring by answer_kind (score() below, tested in tests/test_agent.py):
    number        |predicted - expected| <= tolerance ({"abs": x} or {"rel": x})
    set           exact set match (names normalized, CIKs as integers), plus precision/recall
    entity        a company: entity_cik (or answer_value) equals the expected cik;
                  a fund: answer_value is its fund_id or fund name
    text          citations valid, one cites an expected source prefix, and a keyword hit
    unanswerable  answerable is false

Reports accuracy by question type, citation validity, review rate, average tool calls,
cost per 1,000 questions and latency. Saves agent/results/eval_<target>.csv and
agent/results/examples_<target>.md (8 traces).
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import ML  # noqa: E402

GOLDEN = ML / "data" / "lake" / "golden" / "questions.jsonl"
RESULTS = ML / "agent" / "results"
STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "is", "are", "was",
        "were", "its", "it", "by", "with", "as", "at", "from", "that", "this", "be"}


# ---------- scoring ----------

def _norm(x) -> str:
    """Names compare after the matcher's normalize ('Apple Inc.' == 'APPLE INC');
    numbers compare as their integer text when whole (320193.0 == 320193)."""
    from matching.matcher import normalize

    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return str(int(x)) if float(x).is_integer() else str(x)
    return normalize(str(x))


def _as_number(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return None


def _cik(v):
    if isinstance(v, dict):
        v = v.get("cik")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def keywords(expected) -> list[str]:
    """Keywords to look for: the list itself, or the significant words of a string."""
    if isinstance(expected, list):
        return [str(k).lower() for k in expected]
    words = re.findall(r"[a-z0-9][a-z0-9\-]+", str(expected or "").lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def within(pred: float, want: float, tol) -> bool:
    """tolerance is {"abs": x}, {"rel": x}, a plain number (relative if < 1), or null (exact)."""
    if isinstance(tol, dict):
        allowed = max(float(tol.get("abs") or 0), float(tol.get("rel") or 0) * abs(want))
    elif tol:
        allowed = float(tol) * abs(want) if 0 < float(tol) < 1 else float(tol)
    else:
        allowed = 0.0
    return abs(pred - want) <= allowed + 1e-9 * max(1.0, abs(want))


def _sources(q: dict) -> tuple[list[str], list[str]]:
    """(source prefixes, keywords) from expected_sources, a dict or a plain list."""
    src = q.get("expected_sources") or {}
    if isinstance(src, list):
        return [str(x) for x in src], keywords(q.get("expected"))
    prefixes = list(src.get("chunk_prefixes") or []) + list(src.get("chunk_ids") or [])
    return [str(x) for x in prefixes], ([str(k).lower() for k in src["keywords"]]
                                        if src.get("keywords") else keywords(q.get("expected")))


def score(q: dict, r: dict) -> dict:
    """{"correct": bool, "precision", "recall"} for one question and one agent result."""
    kind, exp, out = q.get("answer_kind"), q.get("expected"), {"precision": None, "recall": None}
    cites = r.get("citations") or []
    value = r.get("answer_value")
    if kind == "number":
        pred, want = _as_number(value), _as_number(exp)
        ok = pred is not None and want is not None and within(pred, want, q.get("tolerance"))
    elif kind == "set":
        pred = value if isinstance(value, list) else [] if value is None else [value]
        p, w = {_norm(x) for x in pred}, {_norm(x) for x in (exp or [])}
        hit = len(p & w)
        out = {"precision": hit / len(p) if p else float(not w),
               "recall": hit / len(w) if w else float(not p)}
        ok = p == w
    elif kind == "entity":
        if isinstance(exp, dict) and exp.get("fund_id"):      # a fund: its id or its name
            names = {_norm(exp["fund_id"]), _norm(exp.get("fund_name", ""))}
            ok = _norm(value) in names if value is not None else False
        else:                                                  # a company: its cik
            want = _cik(exp)
            ok = want is not None and want in (_cik(r.get("entity_cik")), _cik(value))
    elif kind == "text":
        valid = r.get("status") == "done"   # verify passed: every citation is real
        prefixes, kws = _sources(q)
        cited = any(c.startswith(s) for c in cites for s in prefixes) if prefixes else bool(cites)
        text = (r.get("answer") or "").lower()
        ok = valid and cited and any(k in text for k in kws)
    elif kind == "unanswerable":
        ok = r.get("answerable") is False
    else:
        ok = False
    return {"correct": bool(ok)} | out


# ---------- running ----------

def load_golden(path: Path = GOLDEN) -> list[dict]:
    if not path.exists():
        sys.exit(f"no golden questions at {path} (the lake module writes them)")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_one(q: dict, target: str) -> dict:
    t0 = time.perf_counter()
    try:
        if target == "local":
            from agent.graph import ask
            r = ask(q["question"])
        else:
            from agent.invoke import invoke
            r = invoke(q["question"])
    except Exception as e:  # noqa: BLE001: one failure must not stop the evaluation
        r = {"status": "error", "reason": f"{type(e).__name__}: {e}"[:300], "citations": [],
             "trace": []}
    r["latency_s"] = round(time.perf_counter() - t0, 2)
    return r


def summarize(df) -> dict:
    """The headline numbers, over all questions and by type."""
    n = len(df)
    total_cost = df["cost_usd"].fillna(0).sum()
    return {
        "n": n,
        "accuracy": round(df["correct"].mean(), 3),
        "citation_validity": round((df["status"] == "done").mean(), 3),
        "review_rate": round((df["status"] == "needs_human_review").mean(), 3),
        "error_rate": round((df["status"] == "error").mean(), 3),
        "avg_tool_calls": round(df["tool_calls"].fillna(0).mean(), 2),
        "cost_per_1000_usd": round(1000 * total_cost / max(n, 1), 2),
        "latency_p50_s": round(df["latency_s"].median(), 2),
        "latency_p90_s": round(df["latency_s"].quantile(0.9), 2),
    }


def write_examples(rows: list[tuple[dict, dict, dict]], path: Path, n: int = 8) -> None:
    """Eight traces for reading: mix of correct, wrong and reviewed, one per type first."""
    picked, seen = [], set()
    for q, r, s in rows:                      # one per answer kind first
        if q.get("answer_kind") not in seen:
            picked.append((q, r, s))
            seen.add(q.get("answer_kind"))
    for item in rows:                         # then failures, then the rest
        if len(picked) >= n:
            break
        if item not in picked and not item[2]["correct"]:
            picked.append(item)
    for item in rows:
        if len(picked) >= n:
            break
        if item not in picked:
            picked.append(item)
    lines = ["# Agent examples (8 traces)\n",
             "Written by agent/evaluate.py. Each shows the question, what the agent did, "
             "and how it scored.\n"]
    for q, r, s in picked[:n]:
        lines += [f"## {q.get('id')} ({q.get('type')}, {q.get('answer_kind')}): "
                  f"{'correct' if s['correct'] else 'WRONG'}\n",
                  f"**Question:** {q['question']}\n",
                  f"**Expected:** `{json.dumps(q.get('expected'))}`\n",
                  f"**Answer:** {r.get('answer')}  \n**Value:** `{json.dumps(r.get('answer_value'))}`"
                  f"  \n**Status:** {r.get('status')} ({r.get('reason')})  \n"
                  f"**Citations:** {r.get('citations')}\n",
                  "```"]
        for t in r.get("trace", []):
            if t.get("step") == "tool":
                lines.append(f"tool  {t['tool']}({json.dumps(t.get('args'))[:200]}) -> "
                             f"{str(t.get('result'))[:200]}")
            elif t.get("step") == "agent":
                lines.append(f"agent asked for {t.get('asked_for')}")
            else:
                lines.append(f"verify problems={t.get('problems')}")
        lines += ["```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["local", "aws"], default="local")
    parser.add_argument("--limit", type=int, help="first N questions only (no MLflow logging)")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    questions = load_golden()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[:args.limit]
    print(f"{len(questions)} questions, target {args.target}")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda q: run_one(q, args.target), questions))

    rows, records = [], []
    for q, r in zip(questions, results):
        s = score(q, r)
        rows.append((q, r, s))
        records.append({"id": q.get("id"), "type": q.get("type"), "answer_kind": q.get("answer_kind"),
                        "question": q["question"], "expected": json.dumps(q.get("expected")),
                        "answer": r.get("answer"), "answer_value": json.dumps(r.get("answer_value")),
                        "entity_cik": r.get("entity_cik"), "answerable": r.get("answerable"),
                        "citations": json.dumps(r.get("citations")), "status": r.get("status"),
                        "reason": r.get("reason"), "tool_calls": r.get("tool_calls"),
                        "cost_usd": r.get("cost_usd"), "latency_s": r.get("latency_s")} | s)
        mark = "ok " if s["correct"] else "BAD"
        print(f"  {mark} {q.get('id'):>8} {q.get('answer_kind'):>12}  {r.get('status')}  "
              f"{str(r.get('answer'))[:90]!r}")
    df = pd.DataFrame(records)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"eval_{args.target}.csv", index=False)
    write_examples(rows, RESULTS / f"examples_{args.target}.md")  # one per target, never overwritten by the other

    summary = summarize(df)
    by_type = df.groupby("type")["correct"].agg(["mean", "count"]).round(3)
    by_kind = df.groupby("answer_kind")["correct"].agg(["mean", "count"]).round(3)
    print("\naccuracy by type:\n" + by_type.to_string())
    print("\naccuracy by answer kind:\n" + by_kind.to_string())
    print("\n" + "\n".join(f"{k:>20}  {v}" for k, v in summary.items()))
    total = df["cost_usd"].fillna(0).sum()
    print(f"{'llm spend this run':>20}  ${total:.4f}")
    print(f"\nsaved {RESULTS / f'eval_{args.target}.csv'} and {RESULTS / f'examples_{args.target}.md'}")

    if summary["error_rate"] > 0.05:  # a broken run (e.g. the runtime isn't deployed) is not a result
        print(f"error rate {summary['error_rate']:.0%}: not logged to MLflow (fix the target and rerun)")
    elif not args.limit and not args.ids:
        log_mlflow(args.target, summary, by_type, df)


def log_mlflow(target: str, summary: dict, by_type, df) -> None:
    """One MLflow run in experiment "agent", same local store as the other modules."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{ML / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("agent") is None:
        mlflow.create_experiment("agent", artifact_location=(ML / "mlruns").as_uri())
    mlflow.set_experiment("agent")
    from agent.graph import MAX_TOOL_CALLS
    import llm
    with mlflow.start_run(run_name=f"agent-{target}"):
        mlflow.log_params({"target": target, "model": llm.MODEL_ID,
                           "max_tool_calls": MAX_TOOL_CALLS, "n_questions": len(df)})
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if k != "n"})
        mlflow.log_metrics({f"acc_{t}": float(v) for t, v in by_type["mean"].items()})
        mlflow.log_artifact(str(RESULTS / f"eval_{target}.csv"))
        mlflow.log_artifact(str(RESULTS / f"examples_{target}.md"))
    print("logged to MLflow experiment 'agent'")


if __name__ == "__main__":
    main()
