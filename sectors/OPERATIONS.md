# Operating the sector classifier

## 1. Backtesting (done)

Walk-forward by year: train on every year before t, test on year t, for 2017–2020.
- Retrained keeps new-company F1 at 0.73–0.75.
- A model frozen at 2016 falls to 0.65 by 2020.
- A model trained on last year only shows that most of the lift comes from more data, not just
  fresher data.

**Policy:** retrain yearly on all point-in-time history. See `results/backtest.csv` and
`docs/ITERATIONS.md`.

## 2. Precision or recall

**Chosen:** auto-tag at confidence ≥ 0.8 (71% of filings at 86% precision); analysts see the rest.

**To flip:** the `THRESHOLD` environment variable (Lambda in `deploy/aws/sector_stack.py`; Cloud
Run in `deploy/app/main.py`):
- 0.5 → 90% auto-tagged at 80%;
- 0.9 → 60% at 91%.

Change the environment variable and redeploy (`make deploy-aws` / `make deploy-gcp`).

## 3. Cost

| | Idle | Per use |
|---|---|---|
| AWS: API Gateway + Lambda + SageMaker serverless (3 GB) | $0 | API Gateway $3.50 per million requests; Lambda $0.20 per million plus ms of compute; SageMaker serverless billed per ms of compute by memory size (a fraction of a cent per 0.5 s call) |
| Google: Cloud Run (2 vCPU, 4 GiB, min 0) | $0 | per vCPU-second and GiB-second while serving, with a monthly free tier |
| Few-shot LLM instead of the model | $0 | $2.71 per 1,000 filings (measured) |

Measured latency: AWS cold start 24.3 s, warm 0.53 s; Cloud Run 4.3 s, then 0.20 s.

## 4. Monitors

**In place:**
- The Lambda logs sector, route and length per request, never the text.
- SageMaker and API Gateway publish their standard CloudWatch metrics (invocations, latency,
  4xx and 5xx).
- The usage plan meters calls per key.

**Recommended:**
- A metric filter on `route=analyst_review` → review rate; alarm at 1.5× baseline. This is the
  first drift signal.
- Yearly backtest rerun before each retrain.
- Alarms on `Invocation5XXErrors` and `ModelSetupTime` (cold starts).

## 5. Guardrails

**In place:**
- Input 50–100,000 characters, truncated to what the model reads.
- API key with 5 requests per second and 1,000 per day.
- Confidence routing.
- No document text in logs.
- Cloud Run is private behind an IAM identity token.

**Recommended:**
- Real auth instead of the API key (IAM, Cognito or a Lambda authorizer).
- WAF in front of API Gateway.
- Blue/green deploys with automatic rollback on alarms.
