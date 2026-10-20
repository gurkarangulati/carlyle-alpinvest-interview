"""Pick the evaluation sample: ~120 test-split 10-Ks, stratified by sector.

Run:    uv run python extraction/sample.py            (needs fetch_xbrl.py first)
        uv run python extraction/sample.py --n 60     (a smaller sample)

Rules, and why:
  - Test-split companies only, so nothing here was seen while building the
    sector classifier.
  - One filing per company. Two years of the same company look alike, so they
    would count twice for one kind of layout.
  - The filing must have Item 7 or Item 8 text (something to read), and XBRL must
    give at least revenue or net income (something to score against).
  - Stratified by sector with a fixed seed (14): every sector shows up in about
    its real share, and a rerun picks the same filings.
"""

import argparse

from common import SAMPLE_CSV, load_companies, load_texts
from fetch_xbrl import truth_table

MIN_CHARS = 500  # shorter than this is a stub like "See Item 15"

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=120, help="about how many filings")
parser.add_argument("--seed", type=int, default=14)
args = parser.parse_args()

filings = load_companies()
filings = filings[filings["split"] == "test"].drop(columns=["business"])
print(f"test split: {len(filings)} filings from {filings['cik'].nunique()} companies")

truth = truth_table(filings)
filings = filings.merge(truth, on=["cik", "year"])
has_truth = filings["revenue"].notna() | filings["net_income"].notna()
print(f"  with XBRL revenue or net income: {has_truth.sum()}")

texts = load_texts(set(zip(filings["cik"], filings["year"])))
for s in ("section_7", "section_8", "section_15"):
    filings[f"chars_{s[8:]}"] = [len(texts.get(k, {}).get(s, "")) for k in
                                 zip(filings["cik"], filings["year"])]
has_text = (filings["chars_7"] >= MIN_CHARS) | (filings["chars_8"] >= MIN_CHARS)
print(f"  with Item 7 or Item 8 text:      {has_text.sum()}")

eligible = filings[has_truth & has_text]
# One random filing per company, then the same fraction from every sector.
one_each = eligible.groupby("cik").sample(n=1, random_state=args.seed)
frac = min(1.0, args.n / len(one_each))
sample = (one_each.groupby("sector").sample(frac=frac, random_state=args.seed)
          .sort_values(["sector", "cik"]).reset_index(drop=True))

print(f"  eligible: {len(eligible)} filings, {len(one_each)} companies")
print(f"\nsample: {len(sample)} filings")
print(sample["sector"].value_counts().to_string())
print("\nyears:\n" + sample["year"].value_counts().sort_index().to_string())
print(f"\nstatements only in Item 15 (Item 8 is a stub): {(sample['chars_8'] < MIN_CHARS).sum()}")
for f in ("revenue", "net_income", "total_assets", "revenue_prior"):
    print(f"  truth has {f:14s} {sample[f].notna().sum()}")

sample.to_csv(SAMPLE_CSV, index=False)
print(f"saved {SAMPLE_CSV}")
