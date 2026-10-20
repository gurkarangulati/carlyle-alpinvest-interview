"""Split companies into train / val / test, once, and save it.

Every model in this project reads sector_split.json, so they all get
graded on the same companies.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 14

df = pd.read_csv(DATA / "companies.csv")

# One row per company. Split companies, not filings, so a company's
# 2017 and 2018 filings can't end up on both sides (leakage).
companies = df[["cik", "sector"]].drop_duplicates()
assert companies["cik"].is_unique, "a company has more than one sector"

# 70% train, then cut the remaining 30% in half: 15% val, 15% test.
# stratify keeps each pile's sector mix the same as the whole.
train, rest = train_test_split(
    companies, test_size=0.30, stratify=companies["sector"], random_state=SEED
)
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["sector"], random_state=SEED
)

piles = {"train": train, "val": val, "test": test}
split = {name: pile["cik"].tolist() for name, pile in piles.items()}
(DATA / "sector_split.json").write_text(json.dumps(split))

# Report: companies and filings per pile, then sector shares side by side.
for name, ciks in split.items():
    filings = df[df["cik"].isin(ciks)]
    print(f"{name:5}  companies {len(ciks):5,}  filings {len(filings):5,}  "
          f"({len(filings) / len(df):.1%} of filings)")

shares = pd.DataFrame(
    {name: df[df["cik"].isin(ciks)]["sector"].value_counts(normalize=True)
     for name, ciks in split.items()}
)
print("\nSector share of filings in each pile:")
print((shares * 100).round(1).sort_values("train", ascending=False))
