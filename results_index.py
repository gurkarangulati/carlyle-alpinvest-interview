"""Build RESULTS.md: every run of every task, with its numbers, cost, latency and links.

Run:  uv run python results_index.py      (or: make results-index)

Reads MLflow (every run, every experiment) and each module's results folder, then writes:
  RESULTS.md                one page: a table per task, a row per run, links to logs and per-item files
  results/all_runs.csv      every MLflow run with all its parameters and metrics
  */results/per_item/       per-filing / per-question / per-query predictions (latest run)

Links use the public repo's layout (sector outputs live in sectors/results/ there).
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent
CT = ZoneInfo("America/Chicago")
if not (ROOT / "mlflow.db").exists():
    raise SystemExit("No mlflow.db here yet: run the experiments first (make all), then make results-index.")
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
# Sector outputs: sectors/out/ in a working copy, sectors/results/ in the published snapshot.
SECT = "sectors/out" if (ROOT / "sectors" / "out").exists() else "sectors/results"

# Where each module keeps its console logs, here and in the public repo.
LOG_DIRS = {"sectors": (f"{SECT}/logs", "sectors/results/logs"),
            "extraction": ("extraction/results/logs", "extraction/results/logs"),
            "matching": ("matching/results/logs", "matching/results/logs"),
            "lake": ("lake/results/logs", "lake/results/logs"),
            "agent": ("agent/results/logs", "agent/results/logs")}


def public(path: str) -> str:
    """The same file's path in the public repo (sector outputs move, one script was renamed)."""
    return path.replace("sectors/out/", "sectors/results/").replace("llm_zeroshot", "llm_classify")


def link(path: str | None, text: str) -> str:
    return f"[{text}]({public(path)})" if path else "–"


def fmt(x, digits=3, pct=False, money=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    if money:
        return f"${x:,.3f}" if abs(x) < 1 else f"${x:,.2f}"
    return f"{x:.0%}" if pct else f"{x:.{digits}f}"


# ---- logs: match each MLflow run to the console log it printed ----

def log_files(module):
    here, _ = LOG_DIRS[module]
    out = []
    for f in sorted((ROOT / here).glob("*.txt")):
        m = re.match(r"(.*)_(\d{4}-\d{2}-\d{2})_(\d{4})\.txt$", f.name)
        if m:  # stamped at the start of the run
            stem = m.group(1)
            t = datetime.strptime(m.group(2) + m.group(3), "%Y-%m-%d%H%M").replace(tzinfo=CT)
            out.append((stem, t.timestamp(), "start", f"{here}/{f.name}"))
        else:  # tee'd by hand: the file's last write is the end of the run
            out.append((f.stem, f.stat().st_mtime, "end", f"{here}/{f.name}"))
    return out


def find_log(module, stems, start, end, run_name=None):
    """The log this run printed. Prefer logs that say "logged MLflow run <name>" (and, for LLM
    runs, the same pile); break ties by time. A log starts before its MLflow run is written."""
    def time_gap(t, kind):
        return (start - t) if kind == "start" else abs(t - end)

    cands = []
    for stem, t, kind, path in log_files(module):
        if not any(stem == s or stem.startswith(s + "_") for s in stems):
            continue
        gap = time_gap(t, kind)
        if gap < -120 or gap > 3 * 3600:  # log started after the run was logged, or hours before
            continue
        text = (ROOT / path).read_text(errors="ignore")
        strong = False
        if run_name:
            base = run_name.removesuffix("_val")
            names = {base, base.replace("llm_classify", "llm_zeroshot")}
            strong = any(re.search(rf"logged MLflow run {re.escape(n)}\b(?!_)", text) for n in names)
            if base.startswith("llm_"):
                strong = strong and (("--pile val" in text) == run_name.endswith("_val"))
        cands.append((not strong, gap < 0, abs(gap), path))  # a log must start before its run is logged
    return min(cands)[3] if cands else None


def runs(experiment):
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    if df.empty:
        return df
    df["start"] = df["start_time"].map(lambda t: t.timestamp())
    df["end"] = df["end_time"].map(lambda t: t.timestamp() if pd.notna(t) else None)
    df["when"] = df["start_time"].dt.tz_convert(CT).dt.strftime("%m-%d %H:%M")
    return df


def m(row, key):
    v = row.get(f"metrics.{key}")
    return None if v is None or pd.isna(v) else v


# ---- export: every run, every experiment ----

def export_all_runs():
    frames = []
    for e in mlflow.search_experiments():
        df = mlflow.search_runs(experiment_names=[e.name])
        if df.empty:
            continue
        df.insert(0, "experiment", e.name)
        frames.append(df)
    allr = pd.concat(frames, ignore_index=True)
    allr["start_ct"] = allr["start_time"].dt.tz_convert(CT).dt.strftime("%Y-%m-%d %H:%M")
    allr["duration_s"] = (allr["end_time"] - allr["start_time"]).dt.total_seconds().round(1)
    keep = (["experiment", "tags.mlflow.runName", "start_ct", "duration_s", "status"]
            + sorted(c for c in allr.columns if c.startswith("params."))
            + sorted(c for c in allr.columns if c.startswith("metrics.")))
    out = allr[keep].rename(columns={"tags.mlflow.runName": "run"}).sort_values("start_ct")
    out["run"] = out["run"].str.replace("llm_zeroshot", "llm_classify")
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "all_runs.csv", index=False)
    return len(out)


# ---- per-item copies (latest run), committed so every prediction is inspectable ----

def copy_per_item():
    pi = ROOT / "extraction" / "results" / "per_item"
    pi.mkdir(parents=True, exist_ok=True)
    for name in ["preds_baseline", "preds_graph", "qa_preds_baseline", "qa_preds_agent"]:
        src = ROOT / "extraction" / "out" / f"{name}.csv"
        if src.exists():
            shutil.copy(src, pi / src.name)
    pm = ROOT / "matching" / "results" / "per_item"
    pm.mkdir(parents=True, exist_ok=True)
    for split in ["dev", "test"]:  # the full file holds every method; keep the deployed ones
        src = ROOT / "matching" / "out" / f"preds_{split}.csv"
        if src.exists():
            p = pd.read_csv(src)
            p[p["method"].isin(["hybrid", "hybrid_llm"])].to_csv(pm / f"preds_{split}_hybrid.csv",
                                                                index=False)


# ---- sections ----

def sector_preds_file(name, pile):
    """Per-filing predictions for this run name and pile (the latest run of that name)."""
    for n in (name, name.replace("llm_classify", "llm_zeroshot")):
        f = f"{n.removesuffix('_val')}_{pile}_preds.csv"
        if (ROOT / SECT / f).exists():
            return f"{SECT}/{f}"
    return None


def sectors_section():
    df = runs("sectors")
    lines = ["## 1. Sector classification (853 test / 847 validation filings)", "",
             "Macro F1 averages the 10 sectors equally. **$ per 1k correct** = cost per 1,000 "
             "filings ÷ accuracy. Classic models run on a laptop, so their cost is about $0.", "",
             "| When (CT) | Run | Val F1 | Test F1 | Accuracy | $ / 1k filings | $ / 1k correct "
             "| Latency per filing | Train | Log | Per-filing |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        name = r["tags.mlflow.runName"].replace("llm_zeroshot", "llm_classify")
        stems = (["llm_classify", "llm_zeroshot"] if name.startswith("llm_")
                 else ["cv"] if name.endswith("cv5")
                 else ["baseline", "baseline_balanced"] if name.startswith("tfidf")
                 else [name.split("_")[0] if name.startswith("finetune") else name.replace("_walk_forward", "")])
        log = find_log("sectors", stems, r["start"], r["end"], r["tags.mlflow.runName"])
        val, test = m(r, "val_macro_f1"), m(r, "test_macro_f1")
        if val is None and m(r, "cv_mean_f1") is not None:
            val = m(r, "cv_mean_f1")
        acc = m(r, "test_accuracy") or m(r, "val_accuracy")
        cost = m(r, "cost_per_1k_usd")
        lat = m(r, "latency_median_s")
        ms = m(r, "predict_ms_per_1k")
        latency = f"{lat:.2f} s" if lat else (f"{ms / 1000:.3f} ms" if ms else "–")
        pile = "val" if name.endswith("_val") else "test"
        preds = sector_preds_file(name, pile) if (val or test) else None
        lines.append(
            f"| {r['when']} | {name} | {fmt(val)} | {fmt(test)} | {fmt(acc)} "
            f"| {fmt(cost, money=True) if cost else '~$0'} "
            f"| {fmt(cost / acc, money=True) if cost and acc else '~$0' if acc else '–'} "
            f"| {latency} | {fmt(m(r, 'train_seconds'), 0) + ' s' if m(r, 'train_seconds') else '–'} "
            f"| {link(log, 'log')} | {link(preds, 'csv')} |")
    lines += ["", "Per-filing files hold the latest run of each name. The first "
              "`llm_classify_bedrock` test run (0.680) is the scorer bug fixed in the next row; "
              "see `docs/ITERATIONS.md`.", ""]
    bt = ROOT / SECT / "backtest.csv"
    if bt.exists():
        b = pd.read_csv(bt)
        piv = b.pivot(index="test_year", columns="model", values="f1_new")
        lines += ["### Walk-forward backtest: F1 on companies never seen in training", "",
                  "| Test year | Retrained | Frozen (2016) | Last year only | New / seen filings |",
                  "|---|---|---|---|---|"]
        for y, row in piv.iterrows():
            n = b[(b.test_year == y) & (b.model == "retrained")].iloc[0]
            lines.append(f"| {y} | {row['retrained']:.3f} | {row['frozen']:.3f} | "
                         f"{row['last_year']:.3f} | {int(n.n_new)} / {int(n.n_seen)} |")
        lines += ["", "Data: " + " · ".join(link(f"{SECT}/{f}", f) for f in
                  ["backtest.csv", "cv_folds.csv", "split_options.csv", "results.csv"]), ""]
    return lines


def matching_section():
    lines = ["## 2. Company name matching", "",
             "Golden set: SEC former names + generated variants, split by company. Thresholds "
             "are tuned on dev and read by test. **Abstain** = the matcher declined to answer; "
             "**precision when answered** is what an analyst would trust.", ""]
    df = runs("matching")
    lines += ["| When (CT) | Run | Queries | Hybrid top-1 | Precision when answered | Abstain "
              "| Without aliases top-1 | LLM calls / cost | ms per query | Log |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        name = r["tags.mlflow.runName"]
        llm = "llm" in name
        h = "hybrid_llm" if llm else "hybrid"
        stems = ["eval_dev_llm"] if llm else ["eval_" + name.split("_")[1]]
        log = find_log("matching", stems, r["start"], r["end"])
        calls, cost = m(r, "llm_calls"), m(r, "llm_cost_usd")
        lines.append(
            f"| {r['when']} | {name} | {r.get('params.n_queries', '–')} | {fmt(m(r, f'{h}.all.top1'))} "
            f"| {fmt(m(r, f'{h}.all.precision_answered'))} | {fmt(m(r, f'{h}.all.abstain_rate'))} "
            f"| {fmt(m(r, 'tfidf_noalias.all.top1'))} "
            f"| {f'{int(calls)} / ${cost:.2f}' if calls else '–'} "
            f"| {fmt((m(r, f'{h}.all.secs_per_1k_queries') or 0), 3)} | {link(log, 'log')} |")
    t = ROOT / "matching" / "results" / "eval_test.csv"
    if t.exists():
        e = pd.read_csv(t)
        piv = e.pivot(index="type", columns="method", values="top1")
        cols = [c for c in ["exact", "fuzzy", "tfidf", "tfidf_noalias", "hybrid"] if c in piv]
        lines += ["", "### Test top-1 by query type and method", "",
                  "| Query type | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for typ, row in piv.iterrows():
            lines.append(f"| {typ} | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |")
    lines += ["", "Data: " + " · ".join(
        link(f"matching/results/{f}", f) for f in
        ["eval_dev.csv", "eval_test.csv", "hybrid_thresholds.json", "mentions_demo.csv",
         "per_item/preds_dev_hybrid.csv", "per_item/preds_test_hybrid.csv"]
        if (ROOT / "matching" / "results" / f).exists()), ""]
    return lines


def extraction_section():
    df = runs("extraction")
    lines = ["## 3. Number extraction (119 test filings, scored against SEC XBRL)", "",
             "**$ per correct** = run cost ÷ net-income values within 1%. *Not flagged* = "
             "filings the flow did not send to an analyst.", "",
             "| When (CT) | Variant | Revenue ≤1% | Net income ≤1% | Net income ≤1%, not flagged "
             "| Review | Retry | $ / 1k filings | $ per correct net income | Median latency | Log |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    qa = ["", "## 4. Document Q&A (91 questions on 40 filings)", "",
          "Numeric answers scored against XBRL; a right “not in the document” counts as correct. "
          "**$ / 1k correct** = cost per 1,000 questions ÷ accuracy.", "",
          "| When (CT) | Variant | Accuracy ≤5% | Correct when answered | Abstained correctly "
          "| Citations valid | Grounded | Review | Tool calls | $ / 1k questions | $ / 1k correct "
          "| Median / p95 latency | Log |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        name = r["tags.mlflow.runName"]
        if name.startswith("qa_"):
            log = find_log("extraction", [name], r["start"], r["end"], name)
            acc, c1k = m(r, "all_accuracy_5"), m(r, "all_cost_per_1k_usd")
            qa.append(
                f"| {r['when']} | {name.removeprefix('qa_')} | {fmt(acc, pct=True)} "
                f"| {fmt(m(r, 'all_answered_accuracy_5'), pct=True)} "
                f"| {fmt(m(r, 'all_abstained_correctly'), pct=True)} "
                f"| {fmt(m(r, 'all_citation_valid_rate'), pct=True)} | {fmt(m(r, 'all_grounded_rate'), pct=True)} "
                f"| {fmt(m(r, 'all_review_rate'), pct=True)} | {fmt(m(r, 'all_avg_tool_calls'), 1)} "
                f"| {fmt(c1k, money=True)} | {fmt(c1k / acc, money=True) if c1k and acc else '–'} "
                f"| {fmt(m(r, 'all_latency_median_s'), 1)} s / {fmt(m(r, 'all_latency_p95_s'), 1)} s "
                f"| {link(log, 'log')} |")
            continue
        variant = name.removeprefix("extraction_")
        log = find_log("extraction", [f"evaluate_{variant}"], r["start"], r["end"], name)
        n_correct = (m(r, "net_income_acc_1pct") or 0) * (m(r, "net_income_n_truth") or 0)
        cost = m(r, "cost_usd")
        lines.append(
            f"| {r['when']} | {variant} | {fmt(m(r, 'revenue_acc_1pct'), pct=True)} "
            f"| {fmt(m(r, 'net_income_acc_1pct'), pct=True)} "
            f"| {fmt(m(r, 'net_income_acc_1pct_not_flagged'), pct=True)} "
            f"| {fmt(m(r, 'review_rate'), pct=True)} | {fmt(m(r, 'retry_rate'), pct=True)} "
            f"| {fmt(m(r, 'cost_per_1k_usd'), money=True)} "
            f"| {fmt(cost / n_correct, money=True) if cost and n_correct else 'no correct values'} "
            f"| {fmt(m(r, 'latency_median_s'), 1)} s | {link(log, 'log')} |")
    files = ["eval_baseline.csv", "eval_graph.csv", "analyst_notes_sample.md",
             "per_item/preds_baseline.csv", "per_item/preds_graph.csv"]
    lines += ["", "Data: " + " · ".join(link(f"extraction/results/{f}", f) for f in files
                                        if (ROOT / "extraction" / "results" / f).exists()), ""]
    qfiles = ["qa_eval_baseline.csv", "qa_eval_agent.csv", "qa_examples.md",
              "per_item/qa_preds_baseline.csv", "per_item/qa_preds_agent.csv"]
    qa += ["", "Data (per-question rows include the agent's full tool trace): " + " · ".join(
        link(f"extraction/results/{f}", f) for f in qfiles
        if (ROOT / "extraction" / "results" / f).exists()), ""]
    return lines + qa


def folder_section(module, title):
    """Lake and agent: list whatever results and logs exist (they are added after this script)."""
    res = ROOT / module / "results"
    if not res.exists():
        return []
    lines = [f"## {title}", ""]
    for f in sorted(p for p in res.rglob("*") if p.is_file()):
        rel = f"{module}/results/{f.relative_to(res)}"
        lines.append(f"- {link(rel, rel)}")
    return lines + [""]


def all_logs_section():
    lines = ["## Every console log", ""]
    for module, (here, _) in LOG_DIRS.items():
        files = sorted((ROOT / here).glob("*.txt")) if (ROOT / here).exists() else []
        if files:
            lines.append(f"**{module}:** " + " · ".join(link(f"{here}/{f.name}", public(f.stem))
                                                       for f in files))
            lines.append("")
    return lines


if __name__ == "__main__":
    n = export_all_runs()
    copy_per_item()
    page = ["# Results: every run, every task", "",
            "Generated by `results_index.py` (`make results-index`) from MLflow and each "
            "module's results folder. Each row is one run, linked to the console output it printed "
            "and, where kept, its per-item predictions.", "",
            f"- **All {n} runs with every parameter and metric:** {link('results/all_runs.csv', 'results/all_runs.csv')}",
            "- **Browse interactively:** `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` "
            "(after re-running locally; the database itself is not committed)",
            "- **Why each run happened and what it decided:** `docs/ITERATIONS.md`", ""]
    page += sectors_section() + matching_section() + extraction_section()
    page += folder_section("lake", "5. Data lake: build and validation")
    page += folder_section("agent", "6. Lake agent (LangGraph, local and on AgentCore)")
    page += all_logs_section()
    (ROOT / "RESULTS.md").write_text("\n".join(page) + "\n")
    print(f"wrote RESULTS.md and results/all_runs.csv ({n} runs)")
