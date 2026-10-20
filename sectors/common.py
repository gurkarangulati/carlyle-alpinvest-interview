"""Shared helpers: paths, loading the three piles, MLflow setup, scoring.

Every model script imports from here, so all models read the same data,
use the same split, and are scored the same way.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ML = Path(__file__).resolve().parents[1]  # the repo root
DATA = ML / "data"
OUT = Path(__file__).resolve().parent / "out"  # predictions, charts
LOGS = OUT / "logs"  # every run's printed output, one file per run
LOGS.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Send print() output to the screen and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, attr):  # anything else (isatty, encoding...) acts like the screen
        return getattr(self.streams[0], attr)


# Importing common starts the log: out/logs/<script>_<date>_<time>.txt,
# with the exact command at the top.
_log = open(LOGS / f"{Path(sys.argv[0]).stem}_{datetime.now():%Y-%m-%d_%H%M}.txt", "w")
_log.write(f"$ uv run python {' '.join(sys.argv)}\n\n")
sys.stdout = _Tee(sys.__stdout__, _log)


def load_piles() -> dict[str, pd.DataFrame]:
    """Return {"train": rows, "val": rows, "test": rows}, split by company."""
    df = pd.read_csv(DATA / "companies.csv")
    split = json.loads((DATA / "sector_split.json").read_text())
    return {name: df[df["cik"].isin(ciks)].reset_index(drop=True)
            for name, ciks in split.items()}


def start_mlflow() -> None:
    """Log every run to one local database under the experiment "sectors"."""
    mlflow.set_tracking_uri(f"sqlite:///{ML / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("sectors") is None:
        mlflow.create_experiment("sectors", artifact_location=(ML / "mlruns").as_uri())
    mlflow.set_experiment("sectors")


def score(y_true, y_pred) -> dict[str, float]:
    """The two headline numbers every model reports.

    Macro F1 averages over the real sectors only (labels=). Otherwise one off-list
    LLM answer like "software" becomes an 11th sector with F1 0 and drags the average.
    It still counts as a miss for the filing's true sector.
    """
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true))),
        "accuracy": accuracy_score(y_true, y_pred),
    }
