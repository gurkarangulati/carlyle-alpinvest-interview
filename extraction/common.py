"""Shared helpers for the extraction module: paths, loading filings, MLflow setup.

Every script in extraction/ imports from here, so they all read the same files
and log to the same MLflow experiment.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

HERE = Path(__file__).resolve().parent          # extraction/
REPO = HERE.parent
DATA = REPO / "data"
RAW = DATA / "raw" / "edgar"                    # EDGAR-CORPUS jsonl, read only
XBRL_DIR = DATA / "xbrl"                        # one small JSON per company (fetch_xbrl.py)
SAMPLE_CSV = DATA / "extraction_sample.csv"     # the ~120 filings we evaluate on
OUT = HERE / "out"                              # per-filing outputs (git-ignored)
RESULTS = HERE / "results"                      # summary tables kept in git

# The numbers we pull out of each 10-K. The first three are the headline fields;
# the last two are extras the model may return.
FIELDS = ["revenue", "net_income", "total_assets", "operating_income", "cash"]

# The sections we read. Item 8 holds the statements; many small filers write
# "see Item 15" in Item 8 and put the statements at the back, so Item 15 is a fallback.
SECTIONS = ["section_8", "section_7", "section_15"]


def load_companies() -> pd.DataFrame:
    """One row per filing, plus which split (train/val/test) its company is in."""
    df = pd.read_csv(DATA / "companies.csv")
    split = json.loads((DATA / "sector_split.json").read_text())
    df["split"] = None
    for name, ciks in split.items():
        df.loc[df["cik"].isin(ciks), "split"] = name
    return df


def load_texts(keys: set[tuple[int, int]],
               sections: list[str] = SECTIONS) -> dict[tuple[int, int], dict[str, str]]:
    """Read the raw filings for the given (cik, year) pairs.

    The jsonl files are large (about 1.7 GB), so we stream them line by line and
    open only the years we need. Returns {(cik, year): {section_name: text}}.
    """
    found = {}
    for year in sorted({y for _, y in keys}):
        for part in ("test", "validate"):
            with open(RAW / f"{year}_{part}.jsonl") as f:
                for line in f:
                    row = json.loads(line)
                    key = (int(row["cik"]), int(row["year"]))
                    if key in keys and key not in found:
                        found[key] = {s: row.get(s) or "" for s in sections}
    return found


def start_mlflow() -> None:
    """Log every run to one local database under the experiment "extraction"."""
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("extraction") is None:
        mlflow.create_experiment("extraction", artifact_location=(REPO / "mlruns").as_uri())
    mlflow.set_experiment("extraction")
