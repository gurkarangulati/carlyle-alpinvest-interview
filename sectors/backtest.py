"""Walk-forward backtest by year: does the model hold up next year, and does it go stale?

For each test year t (2017-2020), three models, all TF-IDF + logistic regression:
  retrained  - trained on every year before t (what you'd run in production)
  frozen     - trained on 2016 only, never updated
  last_year  - trained on year t-1 only (same size as frozen, but fresh)
frozen vs. last_year isolates staleness: same amount of data, different age.
Each test year is also split into companies the model has seen before vs. new ones.

Run:  uv run python sectors/backtest.py --c 30
"""

import argparse

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import DATA, OUT, score, start_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=30, help="C chosen on val in baseline.py")
args = parser.parse_args()

df = pd.read_csv(DATA / "companies.csv")
years = sorted(df["year"].unique())


def fit(rows: pd.DataFrame):
    """TF-IDF + logistic regression trained on these rows only."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True)
    model = LogisticRegression(C=args.c, max_iter=2000)
    model.fit(vec.fit_transform(rows["business"]), rows["sector"])
    return lambda texts: model.predict(vec.transform(texts))


frozen = fit(df[df["year"] == years[0]])

rows = []
for t in years[1:]:
    past, now = df[df["year"] < t], df[df["year"] == t]
    seen = now["cik"].isin(past["cik"])
    models = {"retrained": fit(past), "frozen": frozen,
              "last_year": fit(df[df["year"] == t - 1])}
    for name, predict in models.items():
        pred = predict(now["business"])
        row = {"test_year": t, "model": name, "train_filings": None,
               "macro_f1": score(now["sector"], pred)["macro_f1"],
               "f1_seen": score(now["sector"][seen], pred[seen])["macro_f1"],
               "f1_new": score(now["sector"][~seen], pred[~seen])["macro_f1"],
               "n_seen": int(seen.sum()), "n_new": int((~seen).sum())}
        row["train_filings"] = {"retrained": len(past), "frozen": (df["year"] == years[0]).sum(),
                                "last_year": (df["year"] == t - 1).sum()}[name]
        rows.append(row)
        print(f"{t}  {name:9}  trained on {row['train_filings']:5,}  "
              f"macro F1 {row['macro_f1']:.3f}  seen {row['f1_seen']:.3f}  "
              f"new {row['f1_new']:.3f}")

results = pd.DataFrame(rows)
results.to_csv(OUT / "backtest.csv", index=False)
print("\nMacro F1 by test year:")
print(results.pivot(index="test_year", columns="model", values="macro_f1").round(3))

start_mlflow()
with mlflow.start_run(run_name="backtest_walk_forward"):
    mlflow.log_params({"model": "tfidf+logreg", "C": args.c, "years": f"{years[0]}-{years[-1]}"})
    for r in rows:
        for k in ["macro_f1", "f1_seen", "f1_new"]:
            mlflow.log_metric(f"{r['model']}_{k}", r[k], step=int(r["test_year"]))
    mlflow.log_artifact(str(OUT / "backtest.csv"))
