"""Precision vs. coverage: what each "dial" buys, measured on the runs already made.

Run:  uv run python precision_curves.py      (or: make precision-curves)

Every task here has a dial between answering more (coverage) and being right when it answers
(precision). This script sweeps each dial over the saved per-item predictions, with no new model
calls, and writes results/precision_coverage.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SECT = ROOT / ("sectors/out" if (ROOT / "sectors" / "out").exists() else "sectors/results")
rows = []


def add(task, dial, setting, covered, correct_covered, n):
    rows.append({"task": task, "dial": dial, "setting": setting, "n": n,
                 "coverage": covered / n if n else None,
                 "precision": correct_covered / covered if covered else None,
                 "correct_of_all": correct_covered / n if n else None})


# 1. Sector tagging: confidence threshold for auto-tagging (the rest go to an analyst).
for name in ["finetune_full", "tfidf_logreg"]:
    p = pd.read_csv(SECT / f"{name}_test_preds.csv")
    right = p["sector"] == p["pred"]
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        keep = p["confidence"] >= t
        add(f"classify ({name})", "confidence ≥", t, keep.sum(), (keep & right).sum(), len(p))

# 2. Name matching: minimum hybrid score before answering (dev split; thresholds are chosen here).
pm = ROOT / "matching" / "results" / "per_item" / "preds_dev_hybrid.csv"
if pm.exists():
    m = pd.read_csv(pm)
    m = m[m["method"] == "hybrid"]
    for t in [0, 50, 60, 70, 80, 90, 95]:
        keep = m["top1_score"] >= t
        add("match (hybrid, dev)", "min score ≥", t, keep.sum(), (keep & m["correct1"]).sum(), len(m))

# 3. Extraction: send failed checks to an analyst, or accept everything.
pe = ROOT / "extraction" / "results" / "per_item" / "preds_graph.csv"
if pe.exists():
    e = pd.read_csv(pe)
    for field in ["revenue", "net_income"]:
        has_truth = e[f"true_{field}"].notna()
        got = e[f"pred_{field}"].notna() & has_truth
        ok = got & ((e[f"pred_{field}"] - e[f"true_{field}"]).abs()
                    <= 0.01 * e[f"true_{field}"].abs())
        n = has_truth.sum()
        add(f"extract ({field})", "policy", "accept every value", got.sum(), ok.sum(), n)
        done = e["status"] == "done"
        add(f"extract ({field})", "policy", "accept only checked (done)",
            (got & done).sum(), (ok & done).sum(), n)

# 4. Document Q&A: which answers to show.
pq = ROOT / "extraction" / "results" / "per_item" / "qa_preds_agent.csv"
if pq.exists():
    q = pd.read_csv(pq)
    q = q[q["kind"] != "text"]  # numeric questions have an answer key
    right = q["correct_5"].astype(bool)
    answered = ~q["abstained"].astype(bool)
    n = len(q)
    add("Q&A agent (numeric)", "policy", "show every answer", answered.sum(), (answered & right).sum(), n)
    add("Q&A agent (numeric)", "policy", "only verified (done)",
        (answered & (q["status"] == "done")).sum(), (answered & right & (q["status"] == "done")).sum(), n)
    add("Q&A agent (numeric)", "policy", "only verified + high confidence",
        (answered & (q["status"] == "done") & (q["confidence"] == "high")).sum(),
        (answered & right & (q["status"] == "done") & (q["confidence"] == "high")).sum(), n)

out = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results" / "precision_coverage.csv", index=False)
with pd.option_context("display.width", 140):
    print(out.round(3).to_string(index=False))
