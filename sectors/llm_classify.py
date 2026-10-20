"""Model 4: zero-shot foundation model. No training: a written sector guide + the filing.

Smoke:  uv run python sectors/llm_classify.py --provider gemini --limit 20
Run:    uv run python sectors/llm_classify.py --provider gemini
        uv run python sectors/llm_classify.py --provider bedrock

Same test filings, same text (first 4,000 chars of Item 1), same score() as every
other model, so the rows in the results table compare fairly.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from common import OUT, load_piles, score, start_mlflow

SECTORS = ["communications_media", "consumer", "energy", "financials", "healthcare",
           "industrials", "materials", "real_estate", "technology", "utilities"]

# The labeling guide, in words. It mirrors the SIC table in fixtures/make_companies.py,
# so the LLM is told the same rules the labels were built from.
GUIDE = """Classify the company into exactly one sector from its 10-K business description.

communications_media: telecom, cable, broadcasting, publishing, film and entertainment, advertising
consumer: food, beverages, tobacco, apparel, furniture, toys, appliances, retail, restaurants,
  hotels, gaming and recreation, personal services, education, agriculture
energy: oil and gas exploration and production, oilfield services, refining, pipelines, coal
financials: banks, lenders, brokers, insurers, asset managers, holding companies
healthcare: pharma, biotech, medical devices, hospitals and clinics, contract research,
  lab and life-science instruments
industrials: construction, machinery, electrical equipment, aerospace, autos and transport
  equipment, industrial instruments, rail/trucking/shipping/airlines/logistics, wholesale
  distribution, business services and staffing, equipment and vehicle rental, engineering
  and consulting
materials: mining, lumber, paper and packaging, chemicals, metals, rubber, plastics, glass
real_estate: real estate operators and developers, REITs
technology: computers and storage, semiconductors, communications equipment, electronic
  test equipment, software and IT services
utilities: electric, gas and water utilities

Answer with the sector only."""

# v2 = v1 plus the tie-break rules already in the SIC table (fixtures/make_companies.py).
# Taken from the labeling spec, not from individual filings; chosen on val, then test once.
GUIDE_V2 = GUIDE.removesuffix("Answer with the sector only.") + """Boundary rules (how these labels are defined):
- Makers of machinery and equipment are industrials whatever their customers' industry
  (oilfield equipment, mining machinery, engines and turbines). Drilling contractors and
  oilfield service operators are energy.
- Equipment rental and leasing companies are industrials.
- Construction and specialty contractors are industrials, even when they build energy or
  utility projects.
- Chemical makers are materials, including ethanol, fertilizer and fuel additives.
- Power generation and distribution, natural gas transmission and distribution, water, and
  waste management are utilities. Crude oil and refined-product pipelines are energy.
- Medical devices and lab instruments are healthcare; industrial measuring and control
  instruments are industrials; electronic test equipment is technology.
- Communications equipment makers are technology; telecom service providers are
  communications_media.
- Retailers are consumer whatever they sell (gas stations, auto dealers, building
  materials). Wholesale distributors are industrials.
- Commercial research labs are healthcare; engineering, consulting and management
  services are industrials.

Answer with the sector only."""
GUIDES = {"v1": GUIDE, "v2": GUIDE_V2}

# List prices, USD per 1M tokens (input, output), checked 2026-10-19. Verify before quoting.
MODELS = {
    "gemini": ("gemini-3.8-flash", 0.75, 3.75),   # Gemini API (AI Studio key, prepaid credits)
    "vertex": ("gemini-3.8-flash", 0.75, 3.75),   # same model via Vertex AI (Cloud Billing, IAM)
    "bedrock": ("us.anthropic.claude-haiku-4-5-20251001-v1:0", 1.00, 5.00),
}

parser = argparse.ArgumentParser()
parser.add_argument("--provider", choices=MODELS, required=True)
parser.add_argument("--pile", choices=["val", "test"], default="test",
                    help="tune prompts on val; open test once")
parser.add_argument("--guide", choices=["v1", "v2"], default="v1",
                    help="v1 = sector list; v2 = plus the SIC table's boundary rules")
parser.add_argument("--shots", type=int, default=0,
                    help="few-shot: add the K most similar TRAIN filings and their labels")
parser.add_argument("--limit", type=int, help="only the first N filings (smoke test)")
parser.add_argument("--workers", type=int, default=8, help="calls in flight at once")
args = parser.parse_args()
model_id, price_in, price_out = MODELS[args.provider]
guide = GUIDES[args.guide]


class Answer(BaseModel):  # structured output: the model can only return one of the 10
    sector: Literal[*SECTORS]


def gemini_caller():
    from google import genai
    from google.genai import types

    if args.provider == "vertex":  # no API key: your gcloud identity (ADC) and IAM decide
        client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                              location="global")
    else:
        client = genai.Client()  # reads GEMINI_API_KEY from the environment
    config = types.GenerateContentConfig(
        system_instruction=guide, temperature=0,
        response_mime_type="application/json", response_schema=Answer,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))

    def call(text):
        r = client.models.generate_content(model=model_id, contents=text, config=config)
        u = r.usage_metadata
        out_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        return r.parsed.sector, u.prompt_token_count, out_tokens

    return call


def bedrock_caller():
    import boto3

    client = boto3.Session(region_name=os.environ.get("AWS_REGION", "us-east-1")).client("bedrock-runtime")
    # Forcing a tool whose input is an enum = structured output on Bedrock.
    tool = {"toolSpec": {"name": "record_sector", "description": "Record the sector.",
                         "inputSchema": {"json": {
                             "type": "object", "required": ["sector"],
                             "properties": {"sector": {"type": "string", "enum": SECTORS}}}}}}

    def call(text):
        r = client.converse(
            modelId=model_id, system=[{"text": guide}],
            messages=[{"role": "user", "content": [{"text": text}]}],
            inferenceConfig={"maxTokens": 50, "temperature": 0},
            toolConfig={"tools": [tool], "toolChoice": {"tool": {"name": "record_sector"}}})
        block = next(c for c in r["output"]["message"]["content"] if "toolUse" in c)
        return (block["toolUse"]["input"]["sector"], r["usage"]["inputTokens"],
                r["usage"]["outputTokens"])

    return call


call = bedrock_caller() if args.provider == "bedrock" else gemini_caller()


def transient(e):
    """Worth retrying? Rate limits (429) and server errors (5xx) clear up; 4xx like
    402 (no credits) or 403 (no access) never will."""
    msg = str(e)
    return any(s in msg for s in ("429", "Throttling", "500", "502", "503", "Unavailable"))


failures_in_a_row = 0
MAX_FAILURES_IN_A_ROW = 10  # circuit breaker: a hard quota won't fix itself, so stop


def classify(text):
    """One filing -> (sector, tokens in, tokens out, seconds). Retries rate limits."""
    global failures_in_a_row
    if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
        return "error", 0, 0, 0.0  # breaker is open: skip the call
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            sector, tin, tout = call(text)
            if sector not in SECTORS:  # never trust the schema alone: check in code
                sector = "invalid"
            failures_in_a_row = 0
            return sector, tin, tout, time.perf_counter() - t0
        except Exception as e:
            if attempt == 3 or not transient(e):
                failures_in_a_row += 1
                print(f"  failed after {attempt + 1} tries: {type(e).__name__}: {str(e)[:120]}")
                if failures_in_a_row == MAX_FAILURES_IN_A_ROW:
                    print(f"  {MAX_FAILURES_IN_A_ROW} failures in a row: stopping the calls")
                return "error", 0, 0, time.perf_counter() - t0
            # Rate limits clear in seconds to a minute, so wait 5, 10, 20 s.
            time.sleep(5 * 2 ** attempt)


piles = load_piles()
rows = piles[args.pile]
if args.limit:
    rows = rows.head(args.limit)
print(f"{args.provider} {model_id}, guide {args.guide}, {args.shots} shots: {len(rows)} "
      f"{args.pile} filings, {args.workers} at a time")

EXAMPLE_CHARS = 600  # the opening of Item 1 is where a company says what it does


def few_shot_prompts(texts):
    """Retrieval few-shot: for each filing, the K most similar TRAIN filings and their labels.

    Similarity = TF-IDF cosine, fit on train only. The examples teach our label
    convention (e.g. oilfield equipment makers are industrials) without writing rules.
    """
    train = piles["train"]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    X_train = vec.fit_transform(train["business"])
    sims = vec.transform(texts) @ X_train.T  # rows are unit length, so this is cosine
    prompts = []
    for i, text in enumerate(texts):
        top = sims[i].toarray().ravel().argsort()[::-1][:args.shots]
        examples = "\n\n".join(
            f"Example {j + 1} (sector: {train['sector'].iat[k]}):\n"
            f"{train['business'].iat[k][:EXAMPLE_CHARS]}" for j, k in enumerate(top))
        prompts.append(f"Labeled examples of similar companies:\n\n{examples}\n\n"
                       f"Now classify this company:\n\n{text}")
    return prompts


texts = list(rows["business"])
prompts = few_shot_prompts(texts) if args.shots else texts

t0 = time.perf_counter()
with ThreadPoolExecutor(args.workers) as pool:  # API calls wait on the network, so overlap them
    results = list(pool.map(classify, prompts))
wall = time.perf_counter() - t0

preds, tok_in, tok_out, secs = map(list, zip(*results))
cost = (sum(tok_in) * price_in + sum(tok_out) * price_out) / 1e6
n = len(rows)
errors = preds.count("error")
metrics = {
    **{f"{args.pile}_{k}": v for k, v in score(rows["sector"], preds).items()},
    "cost_usd": cost,
    "cost_per_1k_usd": cost / n * 1000,
    "latency_median_s": statistics.median(secs),
    "latency_p95_s": sorted(secs)[int(0.95 * (n - 1))],
    "wall_seconds": wall,
    "avg_tokens_in": sum(tok_in) / n,
    "avg_tokens_out": sum(tok_out) / n,
    "errors": errors,
    "invalid_answers": preds.count("invalid"),  # answered, but not one of the 10
}

print(classification_report(rows["sector"], preds, labels=SECTORS, digits=3, zero_division=0))
for k, v in metrics.items():
    print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")

run_name = (f"llm_classify_{args.provider}" + ("" if args.guide == "v1" else f"_{args.guide}")
            + (f"_fewshot{args.shots}" if args.shots else ""))
preds_file = OUT / f"{run_name}_{args.pile}_preds.csv"
pd.DataFrame({"cik": rows["cik"], "year": rows["year"], "true": rows["sector"],
              "pred": preds, "seconds": secs}).to_csv(preds_file, index=False)

if args.limit:
    print("smoke test: not logged to MLflow")
elif errors > 0.05 * n:
    print(f"{errors} of {n} calls failed: not logged to MLflow (fix the quota and rerun)")
else:
    start_mlflow()
    with mlflow.start_run(run_name=f"{run_name}_{args.pile}" if args.pile == "val" else run_name):
        mlflow.log_params({"provider": args.provider, "model": model_id, "pile": args.pile,
                           "guide": args.guide, "shots": args.shots, "n_rows": n,
                           "workers": args.workers, "price_in_per_1m": price_in,
                           "price_out_per_1m": price_out, "text_chars": 4000})
        mlflow.log_metrics(metrics)
        mlflow.log_text(guide, "sector_guide.txt")
        mlflow.log_artifact(str(preds_file))
    print(f"logged MLflow run {run_name}")
print(json.dumps({"saved": str(preds_file)}))
