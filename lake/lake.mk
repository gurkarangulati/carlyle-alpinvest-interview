# The data lake (sprint/ml/lake/): SYNTHETIC fund reports + real 10-Ks -> bronze / silver / gold.
# A fragment for the main Makefile to include (`include lake/lake.mk`); it uses that
# Makefile's $(PY), $(AWS) and check-aws. Run from sprint/ml.  Walkthrough: lake/WALKTHROUGH.md

.PHONY: fund-docs lake-ingest lake-build lake-validate lake-golden lake lake-clean

fund-docs:  ## lake: write the 5 SYNTHETIC fund report PDFs + the truth files
	$(PY) -m lake.funds

lake-ingest:  ## lake: bronze copies of the named companies' 10-K sections and XBRL facts
	$(PY) -m lake.ingest

lake-build: check-aws  ## lake: silver + gold Parquet and DuckDB (extraction on Bedrock, ~$0.01/filing, cached)
	$(AWS) $(PY) -m lake.build

lake-validate:  ## lake: data-quality checks vs. the truth files (exits 1 on a hard fail)
	$(PY) -m lake.validate

lake-golden:  ## lake: ~55 analyst questions with expected answers (from truth + XBRL)
	$(PY) -m lake.golden_questions

lake: fund-docs lake-ingest lake-build lake-validate lake-golden  ## lake: every lake stage, in order

lake-clean:  ## lake: delete data/lake/ (including the cached LLM extraction runs)
	rm -rf data/lake
