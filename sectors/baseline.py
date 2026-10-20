"""Model 1: TF-IDF + logistic regression. The baseline every other model must beat.

Run 1:  uv run python sectors/baseline.py
Run 2:  uv run python sectors/baseline.py --c 10 30 100 300 1000
Run 3:  uv run python sectors/baseline.py --c 10 30 100 300 1000 --balanced
"""

import argparse
import time

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from common import OUT, load_piles, score, start_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, nargs="+", default=[0.3, 1, 3, 10, 30],
                    help="C values to try on val")
parser.add_argument("--balanced", action="store_true",
                    help="weight small sectors up during training")
args = parser.parse_args()
class_weight = "balanced" if args.balanced else None
run_name = "tfidf_logreg" + ("_balanced" if args.balanced else "")

piles = load_piles()
train, val, test = piles["train"], piles["val"], piles["test"]

# 1. Text -> numbers. Learn the vocabulary and word rarity from TRAIN only.
vectorizer = TfidfVectorizer(
    stop_words="english",  # drop "the", "and", "we"
    ngram_range=(1, 2),    # single words and two-word phrases ("natural gas")
    min_df=2,              # ignore words seen in only one filing
    sublinear_tf=True,     # 10 mentions isn't 10x as telling as 1
)
t0 = time.perf_counter()
X_train = vectorizer.fit_transform(train["business"])
X_val = vectorizer.transform(val["business"])
X_test = vectorizer.transform(test["business"])
print(f"{X_train.shape[0]:,} training filings x {X_train.shape[1]:,} word columns")

# 2. Pick C (how freely the weights may grow) using VAL, never test.
sweep = {}
for c in args.c:
    model = LogisticRegression(C=c, max_iter=2000, class_weight=class_weight)
    model.fit(X_train, train["sector"])
    sweep[c] = score(val["sector"], model.predict(X_val))["macro_f1"]
    print(f"  C={c:<6g} val macro F1 {sweep[c]:.3f}")
best_c = max(sweep, key=sweep.get)
if best_c == max(args.c):
    print("  (best C is the largest tried: try bigger values)")

# 3. Refit with the best C, then grade ONCE on test.
model = LogisticRegression(C=best_c, max_iter=2000, class_weight=class_weight)
model.fit(X_train, train["sector"])
train_seconds = time.perf_counter() - t0

t1 = time.perf_counter()
pred = model.predict(X_test)
ms_per_1k = (time.perf_counter() - t1) / len(test) * 1000 * 1000

result = score(test["sector"], pred)
print(f"\nBest C={best_c:g}.  TEST macro F1 {result['macro_f1']:.3f}  "
      f"accuracy {result['accuracy']:.3f}")
print(f"Train {train_seconds:.1f}s total; predict {ms_per_1k:.0f} ms per 1,000 filings\n")
print(classification_report(test["sector"], pred, digits=3))

labels = sorted(train["sector"].unique())
cm = pd.DataFrame(confusion_matrix(test["sector"], pred, labels=labels),
                  index=labels, columns=labels)
print("Confusion matrix (rows = true sector, columns = predicted):")
print(cm.to_string())

# 4. Top words per sector: what the model learned (its weights).
top_words = pd.DataFrame({
    sector: vectorizer.get_feature_names_out()[weights.argsort()[::-1][:8]]
    for sector, weights in zip(model.classes_, model.coef_)
})
print("\nHighest-weight words per sector:")
print(top_words.T.to_string(header=False))

# 5. Save test predictions for error analysis, and log the run.
test.assign(pred=pred, confidence=model.predict_proba(X_test).max(axis=1))[
    ["cik", "year", "name", "sic_description", "sector", "pred", "confidence"]
].to_csv(OUT / f"{run_name}_test_preds.csv", index=False)
cm.to_csv(OUT / f"{run_name}_confusion.csv")
top_words.T.to_csv(OUT / f"{run_name}_top_words.csv")
report = pd.DataFrame(classification_report(test["sector"], pred, output_dict=True)).T
report.to_csv(OUT / f"{run_name}_per_sector.csv")

start_mlflow()
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({"model": "tfidf+logreg", "C": best_c, "C_tried": args.c,
                       "class_weight": class_weight, "ngram": "1-2", "min_df": 2,
                       "vocab": X_train.shape[1]})
    mlflow.log_metrics({**{f"test_{k}": v for k, v in result.items()},
                        "val_macro_f1": sweep[best_c], "train_seconds": train_seconds,
                        "predict_ms_per_1k": ms_per_1k,
                        **{f"val_f1_C_{c:g}": f1 for c, f1 in sweep.items()}})
    for name in ["confusion", "top_words", "per_sector", "test_preds"]:
        mlflow.log_artifact(str(OUT / f"{run_name}_{name}.csv"))
