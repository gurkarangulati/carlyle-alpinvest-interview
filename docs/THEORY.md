# Theory: the ideas behind every choice in this repo

Plain-language explanations, each tied to where it shows up in the code.

## 1. Leakage: the reason most good-looking scores are wrong

**Leakage** means the model sees something at training time that it won't have when it's
used for real. The score then measures memory, not skill.

| Kind | Example here | Guard |
|---|---|---|
| **Row leakage** | Split by filing, and a company's 2017 10-K is in train while its 2018 10-K is in test. 46% of test filings would have their company in train | `split.py` splits by company (cik) |
| **Target leakage** | `sic_description` *is* where the label comes from | The model reads only `business` |
| **Preprocessing leakage** | Fit TF-IDF on all filings, so test vocabulary shapes the features | Fit on train only. In CV, TF-IDF sits inside the `Pipeline`, so each fold refits it |
| **Selection leakage** | Try 10 prompts and report the best one's *test* score | Choose on val; open test once (`llm_classify.py --pile`) |
| **Time leakage** | Train on 2019 and test on 2018 | `backtest.py` trains only on years before t |

## 2. Splits

- **70/15/15 by company.** Train fits the weights, validation chooses the settings, test is
  graded once.
- **Stratify by sector.** Each pile gets each sector's share. Without it, our worst sector
  gap across 20 seeds averaged 3.9 points; with it, 1.5 (`split_options.py`).
- **The seed** fixes the shuffle so the split can be repeated. One seed is one draw.
  `split_options.py` runs 20 seeds to show the spread a single draw hides.
- **StratifiedGroupKFold** keeps companies together *and* balances sectors (gap 0.3). It's
  the production choice and is what `cv.py` uses.

## 3. TF-IDF + logistic regression (the baseline)

- **TF-IDF** turns text into numbers:
  - **TF** is how often a word appears in this filing; `sublinear_tf` dampens repeats.
  - **IDF** is how rare the word is across filings.
  - "Drilling" is rare and telling; "company" is everywhere and tells you nothing.
  - We use single words and two-word phrases ("natural gas"), and drop words seen in only
    one filing.
- **Logistic regression** learns one weight per word per sector. A filing's score for a
  sector is the sum of (word weight × TF-IDF value), and **softmax** turns the 10 scores
  into probabilities.
- **C** is the regularization dial. Small C keeps weights small, which means a simpler and
  more cautious model. Large C lets weights grow to fit the training data, which means more
  detail and more risk of overfitting. C is a *hyperparameter* chosen on validation; the
  weights are *parameters* learned from train.
- **`class_weight="balanced"`** weights each sector's mistakes inversely to its size. It
  raises recall on small sectors at the cost of precision.

## 4. Metrics

- **Precision** for sector X: when the model says X, how often is it right?
- **Recall** for sector X: of all the real X's, how many did it find?
- **F1** is the harmonic mean of the two. It's low if either one is low.
- **Macro F1** averages F1 over the 10 sectors equally, so communications (32 test filings)
  counts as much as industrials (166). That's our headline number. **Accuracy** is reported
  too, but it's dominated by the big sectors.
- **The `labels=` trap.** If an LLM answers "software" once, sklearn's macro average gains
  an 11th class with F1 0. One answer cost 6.8 points before `score()` fixed the label list.

## 5. Cross-validation

One validation pile is one draw. **5-fold CV** trains 5 times, each time holding out a
different fifth of the companies, and reports the mean and the spread across folds. The
C values 30, 100 and 300 scored 0.729, 0.730 and 0.732 with a spread of ±0.014, so the
differences are noise. CV is the *steadier* way to choose; one pile can point to a
"better" C that isn't.

## 6. Backtesting (walk-forward)

The question: **if we had trained the model then, how would it have done the next year?**

- For each test year t (2017–2020), train only on filings from before t and test on year t.
- There are three arms, each tested on the same year-t filings:
  - **Retrained:** all years before t. The production policy.
  - **Frozen:** 2016 only, never updated. The do-nothing policy.
  - **Last year:** t−1 only. This is the same size as frozen but fresher, so it separates
    *freshness* from *amount of data*.
- **Seen vs. new companies.** A returning company scores 0.94–0.98 because the model has
  read its previous 10-K. That's memory. The honest number for a new portfolio company is
  the new-company F1.
- **Drift / staleness:** performance falls as the world moves away from the training data
  (new business models, new words). The frozen arm measures it: 8 points down on new
  companies by 2020.
- **Point-in-time** in practice: train only on what was known at the time, including
  companies that later disappeared. Dropping those creates survivorship bias.

## 7. Fine-tuning a transformer

- **DistilRoBERTa** is a pretrained language model with 82M weights. We add a 10-way
  classification head and keep training on our labels.
- **The tokenizer** splits text into sub-word tokens; we keep the first 256 (about 1,000
  characters).
- **The Trainer loop:**
  - **Epochs** are passes over the training data.
  - **Learning rate** is the step size: 5e-5 for full fine-tuning, larger for LoRA.
  - **Warmup** ramps the step size up at the start.
  - **Weight decay** shrinks weights a little each step, a regularizer.
  - **Evaluation each epoch** with `load_best_model_at_end` keeps the best epoch.
- **Overfitting signal:** validation loss rising while training loss falls. It appeared at
  epoch 3.
- **LoRA** freezes the 82M weights and learns small low-rank corrections in the attention
  layers: rank 8, alpha 16, on the query and value matrices.
  - Only 0.9% of the weights train, and the adapter file is 3 MB.
  - One base model can serve many adapters.
  - `merge_and_unload` folds the adapter back into the base weights for serving.

## 8. Foundation models: zero-shot, rules, few-shot

- **Zero-shot:** no training. The model gets a written sector guide and the filing.
- **Rules (`--guide v2`):** the guide plus tie-break rules copied from the labeling spec,
  for example "makers of oilfield equipment are industrials".
- **Few-shot with retrieval (`--shots 5`):**
  - Each prompt also carries the 5 most similar *training* filings with their labels,
    found by TF-IDF cosine similarity.
  - The model learns the label convention from examples, in the companies' own words.
  - The examples come from train only; test labels never appear in a prompt.
- **Structured output:**
  - Gemini constrains its output to the schema as it generates.
  - Bedrock's forced tool call is guidance, and one answer in 853 came back off-list.
  - **Check every answer in code.**
- **Cost per task, not per token.** Tokenizers differ, the tool schema adds tokens, and
  "thinking" models add hidden output tokens. Compare cost per 1,000 filings.
- **What labels mean.** Under SIC our labels put oilfield equipment makers in industrials;
  under GICS, the investor standard, they're Energy. Some LLM "errors" are the
  investor-correct answer. **Which taxonomy to use is a product decision.**

## 9. Serving

- **Confidence routing:** if the top probability is at least the threshold, the item is
  auto-tagged; otherwise it goes to an analyst.
  - The threshold is a business dial: 0.8 auto-tags 71% of filings at 86% accuracy, and
    0.9 auto-tags 60% at 91%.
- **Serverless vs. always-on.** Serverless costs $0 when idle but has cold starts (24 s
  measured on SageMaker). An always-on instance is always warm but bills every hour.
- **SageMaker vs. Bedrock.**
  - **SageMaker hosts *your* model**: our fine-tuned classifier.
  - **Bedrock is an API to *providers'* foundation models**: Claude, Llama, Mistral and
    others. On Google, Gemini lives in Vertex AI. The same split applies there: Cloud Run
    or Vertex endpoints for your model, the Gemini API on Vertex for theirs.
- **API keys meter usage; they don't authenticate.** For real access control, use IAM, as
  Cloud Run does with identity tokens, or Cognito or a Lambda authorizer.

## 10. Entity resolution (`matching/`)

- **Normalize first:** case, punctuation, legal suffixes ("Inc", "Incorporated").
- **Aliases:** former names, each with the dates it was valid, let "as of" lookups work.
- **Candidates and scoring:**
  - Character n-gram TF-IDF finds candidates and survives typos.
  - Fuzzy string scores re-rank them.
- **Abstain** when the top score or the margin over the runner-up is low. A wrong match is
  worse than "I don't know".
- See `matching/WALKTHROUGH.md`.

## 11. Extraction with checks (`extraction/`)

- **XBRL** is the SEC's tagged version of the financial statements, so it's an answer key.
- **A graph, not one prompt:**
  - locate → extract → normalize → check;
  - one retry with a different text window when a check fails;
  - a **human-review route** with a reason.
- **Checks in code:**
  - the evidence quote must appear in the text;
  - scale ("in thousands") is applied explicitly;
  - plausibility limits;
  - an analyst note may only cite computed numbers.
- See `extraction/WALKTHROUGH.md`.
