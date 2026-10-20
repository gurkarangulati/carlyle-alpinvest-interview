# Samples

Committed copies of what `make fund-docs` and `make lake-golden` generate, so the documents and labels can be read without rebuilding.

- `fund_reports/F01.pdf`..`F05.pdf`: the five SYNTHETIC fund reports the lake is built from.
- `fund_truth.csv`, `funds_truth.csv`: the labels (true company per reported name, true fund figures). The lake never reads these; only validation and the golden set do.
- `questions.jsonl`: the 57 golden analyst questions with expected answers.
