"""Print every logged run as one table, and save it to out/results.csv.

Run:  uv run python sectors/report.py
"""

import mlflow

from common import OUT, start_mlflow

start_mlflow()
runs = mlflow.search_runs(experiment_names=["sectors"], order_by=["start_time ASC"])
cols = {"start_time": "when", "tags.mlflow.runName": "run",
        "params.C": "C", "params.C_tried": "C_tried", "params.class_weight": "class_weight",
        "metrics.val_macro_f1": "val_f1", "metrics.cv_mean_f1": "cv_mean_f1",
        "metrics.cv_std_f1": "cv_spread", "metrics.test_macro_f1": "test_f1",
        "metrics.test_accuracy": "test_acc", "metrics.train_seconds": "train_s",
        "metrics.predict_ms_per_1k": "ms_per_1k",
        "metrics.cost_per_1k_usd": "usd_per_1k", "metrics.latency_median_s": "llm_median_s",
        "metrics.errors": "llm_errors"}
table = runs[[c for c in cols if c in runs]].rename(columns=cols)
table["when"] = table["when"].dt.tz_convert("America/Chicago").dt.strftime("%m-%d %H:%M")
print(table.round(3).to_string(index=False))
table.to_csv(OUT / "results.csv", index=False)
