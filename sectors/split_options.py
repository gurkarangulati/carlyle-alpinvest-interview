"""Compare four ways to split companies 70/15/15, over 20 different seeds.

"Worst gap" = for the most uneven sector, how far apart its share of filings is
between the piles (in percentage points). Smaller = more even piles.
Running 20 seeds shows how much each method depends on the luck of the shuffle.

Run:  uv run python sectors/split_options.py
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import DATA, OUT, start_mlflow

df = pd.read_csv(DATA / "companies.csv")
companies = df[["cik", "sector"]].drop_duplicates()
n_filings = df.groupby("cik").size()
companies["sector_x_count"] = (companies["sector"] + "_"
                               + np.where(companies["cik"].map(n_filings) > 1, "2plus", "1yr"))


def two_cuts(stratify_col, seed):
    """Our method: 70/30, then 30 -> 15/15. stratify_col=None means plain random."""
    strat = companies[stratify_col] if stratify_col else None
    train, rest = train_test_split(companies, test_size=0.30, stratify=strat, random_state=seed)
    strat = rest[stratify_col] if stratify_col else None
    val, test = train_test_split(rest, test_size=0.50, stratify=strat, random_state=seed)
    return [train["cik"], val["cik"], test["cik"]]


def group_kfold(seed):
    """StratifiedGroupKFold into 20 slices of 5%: 14 train, 3 val, 3 test."""
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    slices = [df["cik"].iloc[idx] for _, idx in sgkf.split(df, df["sector"], groups=df["cik"])]
    return [pd.concat(slices[:14]), pd.concat(slices[14:17]), pd.concat(slices[17:])]


def measure(piles):
    shares = pd.DataFrame({i: df[df["cik"].isin(p)]["sector"].value_counts(normalize=True)
                           for i, p in enumerate(piles)})
    worst_gap = (shares.max(axis=1) - shares.min(axis=1)).max() * 100
    sizes = [df["cik"].isin(p).mean() * 100 for p in piles]
    return worst_gap, sizes


methods = {
    "no stratify (plain random)": lambda s: two_cuts(None, s),
    "stratify by sector (ours)": lambda s: two_cuts("sector", s),
    "stratify by sector + filing count": lambda s: two_cuts("sector_x_count", s),
    "StratifiedGroupKFold (20 x 5%)": group_kfold,
}
rows = []
for name, make in methods.items():
    gaps = [measure(make(seed))[0] for seed in range(1, 21)]
    gap14, sizes14 = measure(make(14))
    rows.append({"method": name, "filings % (seed 14)": "/".join(f"{s:.1f}" for s in sizes14),
                 "worst gap, seed 14": round(gap14, 1),
                 "worst gap, avg of 20 seeds": round(np.mean(gaps), 1),
                 "worst gap, range of 20 seeds": f"{min(gaps):.1f}-{max(gaps):.1f}"})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(OUT / "split_options.csv", index=False)

start_mlflow()
with mlflow.start_run(run_name="split_options"):
    mlflow.log_params({"seeds": "1-20", "piles": "70/15/15"})
    mlflow.log_artifact(str(OUT / "split_options.csv"))
