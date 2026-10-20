"""Round: choose C by 5-fold cross-validation instead of one val pile.

Each C is scored 5 times (each fold of companies takes a turn as the held-out
part), so we see the average AND the spread. A small spread = a steadier choice.
Test stays locked until the end, same as baseline.py.

Run:  uv run python sectors/cv.py --c 10 30 100 300 1000
"""

import argparse
import time

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from common import OUT, load_piles, score, start_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, nargs="+", default=[10, 30, 100, 300, 1000])
args = parser.parse_args()

piles = load_piles()
trainval = pd.concat([piles["train"], piles["val"]], ignore_index=True)  # 85%
test = piles["test"]

# TF-IDF goes INSIDE the pipeline, so each fold learns its vocabulary only from
# its own training part. Fitting TF-IDF once on all 85% would leak.
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True)),
    ("logreg", LogisticRegression(max_iter=2000)),
])
search = GridSearchCV(
    pipe,
    {"logreg__C": args.c},                     # the C values to try
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=14),
    scoring="f1_macro",
    n_jobs=-1,                                 # run folds in parallel
)
t0 = time.perf_counter()
search.fit(trainval["business"], trainval["sector"], groups=trainval["cik"])
fit_seconds = time.perf_counter() - t0

cv = pd.DataFrame(search.cv_results_)
folds = cv[[f"split{i}_test_score" for i in range(5)]].round(3)
folds.columns = [f"fold{i + 1}" for i in range(5)]
folds.insert(0, "C", cv["param_logreg__C"])
folds["mean"] = cv["mean_test_score"].round(3)
folds["spread(std)"] = cv["std_test_score"].round(3)
print("Macro F1 on each held-out fold, for each C:")
print(folds.to_string(index=False))

best_c = search.best_params_["logreg__C"]
best = cv.loc[search.best_index_]
pred = search.predict(test["business"])  # best C, refit on all 85%
result = score(test["sector"], pred)
print(f"\nBest C={best_c:g}: CV mean {best['mean_test_score']:.3f} "
      f"+/- {best['std_test_score']:.3f}")
print(f"TEST macro F1 {result['macro_f1']:.3f}  accuracy {result['accuracy']:.3f}  "
      f"({fit_seconds:.0f}s for {len(args.c) * 5} fits + refit)")

folds.to_csv(OUT / "cv_folds.csv", index=False)
start_mlflow()
with mlflow.start_run(run_name="tfidf_logreg_cv5"):
    mlflow.log_params({"model": "tfidf+logreg", "C": best_c, "C_tried": args.c,
                       "selection": "5-fold StratifiedGroupKFold on train+val"})
    mlflow.log_metrics({**{f"test_{k}": v for k, v in result.items()},
                        "cv_mean_f1": best["mean_test_score"],
                        "cv_std_f1": best["std_test_score"], "train_seconds": fit_seconds})
    mlflow.log_artifact(str(OUT / "cv_folds.csv"))
