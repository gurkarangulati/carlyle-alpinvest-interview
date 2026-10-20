"""Shared helpers: paths, loading the name universe, MLflow setup.

Every matching script imports from here, so they all read the same files
and log to the same MLflow experiment.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
HERE = Path(__file__).resolve().parent  # matching/
RESULTS = HERE / "results"  # small summary tables, kept in git
OUT = HERE / "out"  # per-query predictions, git-ignored
RESULTS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEC_NAMES = DATA / "sec_names.jsonl"  # one line per CIK in companies.csv, with former names
SEC_TICKERS = DATA / "company_tickers.json"  # every listed SEC filer: the wider universe
GOLDEN = DATA / "match_golden.csv"
THRESHOLDS = RESULTS / "hybrid_thresholds.json"  # chosen on dev by evaluate.py


def load_universe() -> list[dict]:
    """Return one record per CIK: {cik, name, former_names, tickers}.

    Two sources, merged by CIK:
    - sec_names.jsonl: our 10-K companies, with SEC's history of former names.
    - company_tickers.json: ~10k currently listed filers (current name and ticker only).
    The second one makes the haystack bigger, so a match has real competition.
    """
    companies = {}
    for line in SEC_NAMES.read_text().splitlines():
        r = json.loads(line)
        companies[r["cik"]] = {"cik": r["cik"], "name": r["name"],
                               "former_names": r["formerNames"], "tickers": r["tickers"]}
    for t in json.loads(SEC_TICKERS.read_text()).values():
        c = companies.setdefault(int(t["cik_str"]), {"cik": int(t["cik_str"]), "name": t["title"],
                                                     "former_names": [], "tickers": []})
        if t["ticker"] not in c["tickers"]:
            c["tickers"].append(t["ticker"])
    return list(companies.values())


def start_mlflow() -> None:
    """Log every run to one local database under the experiment "matching"."""
    import mlflow  # imported here: it is slow to load, and only evaluate.py needs it

    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("matching") is None:
        mlflow.create_experiment("matching", artifact_location=(REPO / "mlruns").as_uri())
    mlflow.set_experiment("matching")
