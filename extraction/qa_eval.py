"""Evaluate the Q&A agent against a single-call baseline, on the same questions.

Smoke:  AWS_PROFILE=<your-profile> uv run python extraction/qa_eval.py --variant agent --limit 6
        AWS_PROFILE=<your-profile> uv run python extraction/qa_eval.py --variant baseline --limit 6
Run:    AWS_PROFILE=<your-profile> uv run python extraction/qa_eval.py --variant baseline
        AWS_PROFILE=<your-profile> uv run python extraction/qa_eval.py --variant agent

Questions (built from data/extraction_sample.csv, seed 14):
  (a) numeric, with an XBRL answer key, for 40 filings that have XBRL revenue:
      "What was total revenue in fiscal {year}?" and, where XBRL has the prior year,
      "By what percent did total revenue change in fiscal {year} versus the prior year?"
  (b) text, for 5 of those filings: main risks, largest customers, number of employees.
      There is no answer key for these. We score only what code can check: are the
      citations valid, and is every number grounded in a cited chunk. That says the
      answer is sourced, not that it is right.

Scoring the numeric questions, abstention included:
  - "in_text": does the true number appear in the filing text at all (a dollar amount
    within 1% of it)? For growth: are both years' revenue in the text?
  - An answer is correct if its number is within 1% (or 5%) of the truth. Growth is
    scored in percentage points (within 1 or 5 points), since "12% vs 12.6%" is
    really a 0.6-point gap.
  - Saying "not in the document" is CORRECT when the number is not in the text, and
    wrong when it is. A system that guesses is worse than one that abstains.

(c) baseline: one LLM call on the first 12,000 characters of Item 7, no tools, the same
    final_answer schema. Its citations and numbers go through the same code checks
    (reported, but it has no review route).

The agent's get_financials tool reads the extraction graph's saved outputs. Run
extraction/evaluate.py --variant graph first, or the agent runs extraction live.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm
import mlflow
import pandas as pd
from common import OUT, RESULTS, SAMPLE_CSV, load_texts, start_mlflow
from qa_agent import (
    NOT_FOUND,
    SYSTEM,
    TOOLS,
    answer_question,
    clean_final,
    verify_answer,
)
from qa_scoring import dollar_amounts, grade, in_text, predicted_value
from qa_tools import QA_SECTIONS, filing_index, make_chunks, use_texts

TEXT_QUESTIONS = [
    "What are the main risk factors the company describes?",
    "Who are the company's largest customers, or how concentrated is its customer base?",
    "How many employees does the company have?",
]
BASELINE_CHARS = 12_000

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["agent", "baseline"], required=True)
parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
parser.add_argument("--filings", type=int, default=40, help="filings for numeric questions")
parser.add_argument("--text-filings", type=int, default=5, help="filings for text questions")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()


# ---------- questions ----------

def make_questions(sample: pd.DataFrame) -> list[dict]:
    """Per filing: revenue, then growth (if prior year known), then the text questions."""
    pool = sample[sample["revenue"].notna()]
    pool = pool.sample(n=min(args.filings, len(pool)), random_state=14)
    questions = []
    for i, r in enumerate(pool.itertuples()):
        base = {"cik": int(r.cik), "year": int(r.year), "name": r.name, "sector": r.sector}
        questions.append(base | {"kind": "revenue", "truth": r.revenue,
                                 "question": f"What was total revenue in fiscal {r.year}?"})
        if pd.notna(r.revenue_prior) and r.revenue_prior > 0:
            questions.append(base | {
                "kind": "growth", "truth": (r.revenue / r.revenue_prior - 1) * 100,
                "prior": r.revenue_prior, "revenue": r.revenue,
                "question": f"By what percent did total revenue change in fiscal {r.year} "
                            "versus the prior year?"})
        if i < args.text_filings:
            questions += [base | {"kind": "text", "truth": None, "question": q}
                          for q in TEXT_QUESTIONS]
    return questions


# ---------- the two variants ----------

def run_agent(q: dict) -> dict:
    state = answer_question(q["cik"], q["year"], q["question"])
    return {"final": clean_final(state.get("final")), "status": state.get("status"),
            "reason": state.get("reason"), "problems": state.get("problems", []),
            "tool_calls": state.get("tool_calls", 0), "revisions": state.get("revisions", 0),
            "tokens_in": state.get("tokens_in", 0), "tokens_out": state.get("tokens_out", 0),
            "trace": state.get("trace", []), "path": state.get("path", []),
            "retrieved": state.get("retrieved", {})}


BASELINE_SYSTEM = f"""Answer the analyst's question about a 10-K using only the excerpt.
The excerpt is split into chunks, each starting with its [chunk_id]. Call final_answer.
Cite the chunk_ids that support the answer, and copy numbers as written in them.
If the excerpt does not contain the answer, set answerable to false and answer
"{NOT_FOUND}" Do not guess."""
FINAL_TOOL = next(t for t in TOOLS if t["name"] == "final_answer")


def run_baseline(q: dict) -> dict:
    sections = {"section_7": TEXTS.get((q["cik"], q["year"]), {}).get("section_7", "")}
    chunks = {i: t for i, t in make_chunks(sections).items()
              if int(i.split("-")[1]) * 1500 < BASELINE_CHARS}
    excerpt = "\n\n".join(f"[{i}]\n{t}" for i, t in chunks.items())
    r = llm.converse(BASELINE_SYSTEM, f"Question: {q['question']}\n\nExcerpt:\n\n{excerpt}",
                     tool=FINAL_TOOL, max_tokens=800)
    known = set(filing_index(q["cik"], q["year"])[0])
    problems = verify_answer(r["output"], q["question"], chunks, known, [], [])
    return {"final": clean_final(r["output"]), "status": "done", "reason": "baseline: no route",
            "problems": problems, "tool_calls": 0, "revisions": 0,
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "trace": [], "path": ["single_call"], "retrieved": chunks}


def run_one(q: dict) -> dict:
    t0 = time.perf_counter()
    try:
        out = (run_agent if args.variant == "agent" else run_baseline)(q)
    except Exception as e:  # noqa: BLE001 (e.g. circuit breaker open): record it, keep going
        out = {"final": {}, "status": "error", "reason": f"{type(e).__name__}: {str(e)[:150]}",
               "problems": ["error"], "tokens_in": 0, "tokens_out": 0}
    return q | out | {"seconds": time.perf_counter() - t0}


# ---------- scoring one question ----------

def score(r: dict) -> dict:
    final = r.get("final") or {}
    answer = str(final.get("answer") or "")
    abstained = final.get("answerable") is False
    problems = r.get("problems") or []
    out = {
        "answer": answer, "answerable": final.get("answerable"),
        "confidence": final.get("confidence"), "citations": final.get("citations"),
        "abstained": abstained,
        "citation_valid": bool(final) and not any("citation" in p for p in problems),
        "grounded": bool(final) and not any("numbers not" in p or "calculation" in p
                                            for p in problems),
    }
    if r["kind"] == "text":
        return out
    sections = TEXTS.get((r["cik"], r["year"]), {})
    amounts = dollar_amounts(" ".join(sections.values()))
    if r["kind"] == "revenue":
        findable = in_text(amounts, r["truth"])
    else:
        findable = in_text(amounts, r["revenue"]) and in_text(amounts, r["prior"])
    pred = None if abstained else predicted_value(r["kind"], answer)
    err = None
    for tol in (1, 5):
        out[f"correct_{tol}"], err = grade(r["kind"], pred, r["truth"], abstained, findable, tol)
    return out | {"in_text": findable, "pred_value": pred, "error": err}


# ---------- run ----------

sample = pd.read_csv(SAMPLE_CSV)
questions = make_questions(sample)
if args.limit:
    questions = questions[: args.limit]
keys = {(q["cik"], q["year"]) for q in questions}
print(f"{args.variant}: {len(questions)} questions on {len(keys)} filings, model {llm.MODEL_ID}")
print("loading filing texts...")
TEXTS = load_texts(keys, list(QA_SECTIONS))
use_texts(TEXTS)

results = []
t0 = time.perf_counter()
with ThreadPoolExecutor(args.workers) as pool:
    futures = [pool.submit(run_one, q) for q in questions]
    for i, fut in enumerate(as_completed(futures), 1):
        r = fut.result()
        results.append(r | score(r))
        elapsed = time.perf_counter() - t0
        print(f"  [{i}/{len(futures)}] {r['cik']} {r['kind']:7s} {r['status']:18s} "
              f"{r['seconds']:5.1f}s  elapsed {elapsed:.0f}s  eta "
              f"{elapsed / i * (len(futures) - i):.0f}s")
wall = time.perf_counter() - t0
df = pd.DataFrame(results)


def summarize(g: pd.DataFrame, kind: str) -> dict:
    numeric = g[g["kind"] != "text"]
    secs = sorted(g["seconds"])
    cost = llm.cost(g["tokens_in"].sum(), g["tokens_out"].sum())
    row = {"kind": kind, "n": len(g),
           "abstention_rate": g["abstained"].mean(),
           "citation_valid_rate": g["citation_valid"].mean(),
           "grounded_rate": g["grounded"].mean(),
           "review_rate": (g["status"] == "needs_human_review").mean(),
           "error_rate": (g["status"] == "error").mean(),
           "avg_tool_calls": g["tool_calls"].fillna(0).mean(),
           "cost_usd": cost, "cost_per_1k_usd": cost / len(g) * 1000,
           "latency_median_s": statistics.median(secs),
           "latency_p95_s": float(pd.Series(secs).quantile(0.95))}
    if len(numeric):
        answered = numeric[~numeric["abstained"]]
        abst = numeric[numeric["abstained"]]
        row |= {"accuracy_1": numeric["correct_1"].mean(), "accuracy_5": numeric["correct_5"].mean(),
                "in_text_rate": numeric["in_text"].mean(),
                "answered_accuracy_5": answered["correct_5"].mean() if len(answered) else None,
                "abstained_correctly": (abst["in_text"] == False).mean() if len(abst) else None,
                "accuracy_5_not_flagged":
                    numeric.loc[numeric["status"] == "done", "correct_5"].mean()}
    return row


summary = pd.DataFrame([summarize(g, k) for k, g in df.groupby("kind")] + [summarize(df, "all")])
pd.set_option("display.width", 200)
print("\n" + summary.round(3).T.to_string(header=False))

out_dir = OUT / "smoke" if args.limit else OUT
results_dir = out_dir if args.limit else RESULTS
out_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
preds_file = out_dir / f"qa_preds_{args.variant}.csv"
cols = ["cik", "year", "name", "kind", "question", "truth", "in_text", "answer", "pred_value",
        "error", "correct_1", "correct_5", "abstained", "answerable", "confidence", "citations",
        "citation_valid", "grounded", "status", "reason", "problems", "tool_calls",
        "revisions", "tokens_in", "tokens_out", "seconds", "path", "trace"]
df.reindex(columns=cols).to_csv(preds_file, index=False)
eval_file = results_dir / f"qa_eval_{args.variant}.csv"
summary.to_csv(eval_file, index=False)


# ---------- examples: full traces, good and bad ----------

def write_examples(path, k: int = 8) -> None:
    good = df[(df["status"] == "done") & (df["correct_5"].fillna(True))]
    bad = df[(df["status"] != "done") | (df["correct_5"] == False)]
    picks = pd.concat([good.head(k // 2), bad.head(k - min(k // 2, len(good)))])
    lines = ["# Q&A agent: example traces", "",
             "Picked from the evaluation run: the first answers that passed and were right",
             "(or text answers that passed the checks), then the first that were wrong or sent",
             "to review. Text answers have no answer key.", ""]
    for _, r in picks.iterrows():
        truth = "" if r["kind"] == "text" else f"  \nXBRL truth: {r['truth']:,.2f} " \
            f"(in the text: {r['in_text']}; correct within 5: {r['correct_5']})"
        lines += [f"## {r['name']} ({r['year']}): {r['question']}", "",
                  f"Status: **{r['status']}** ({r['reason']}){truth}", "",
                  f"> {r['answer']}", "",
                  f"Citations: {r['citations']}  ", f"Path: {' > '.join(r['path'] or [])}", "",
                  "Trace:", ""]
        lines += [f"{i}. `{t['tool']}({json.dumps(t['args'])})` -> {t['result'][:200]}"
                  for i, t in enumerate(r["trace"] or [], 1)] or ["(no tool calls)"]
        lines.append("")
    path.write_text("\n".join(lines))


examples_file = results_dir / "qa_examples.md"
if args.variant == "agent":
    write_examples(examples_file)

# ---------- MLflow ----------

errors = int((df["status"] == "error").sum())
if args.limit:
    print("\nsmoke test: not logged to MLflow")
elif errors > 0.05 * len(df):
    print(f"\n{errors} of {len(df)} questions errored: not logged to MLflow")
else:
    start_mlflow()
    with mlflow.start_run(run_name=f"qa_{args.variant}"):
        mlflow.log_params({"variant": args.variant, "model": llm.MODEL_ID,
                           "n_questions": len(df), "filings": args.filings,
                           "text_filings": args.text_filings, "workers": args.workers})
        for r in summary.to_dict("records"):
            mlflow.log_metrics({f"{r['kind']}_{k}": float(v) for k, v in r.items()
                                if k != "kind" and v is not None and pd.notna(v)})
        mlflow.log_metric("wall_seconds", wall)
        mlflow.log_text(SYSTEM if args.variant == "agent" else BASELINE_SYSTEM, "system_prompt.txt")
        for f in (eval_file, preds_file) + ((examples_file,) if args.variant == "agent" else ()):
            mlflow.log_artifact(str(f))
    print(f"\nlogged MLflow run qa_{args.variant}")
print(json.dumps({"summary": str(eval_file), "per_question": str(preds_file)}))
