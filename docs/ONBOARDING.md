# Onboarding: everything set up once before the runbook works

`sectors/README.md` is the runbook (what to run, in order). This file is the one-time setup behind it:
accounts, permissions, quotas and logins, in the order you'll hit them, plus every error we
hit while building this and its fix. Run every command from the repo root unless it says otherwise.

## 0. Tools on the laptop

| Tool | Why | Check |
|---|---|---|
| `uv` | Python, packages, one locked environment | `uv --version` |
| Node.js (for `npx`) | The CDK command-line tool is a Node package | `node --version` |
| AWS CLI | Credentials, quotas, checking resources | `aws --version` |
| `gcloud` | Google Cloud login, APIs, deploys | `gcloud --version` |

Python packages:
- `uv sync` installs the core packages: pandas, scikit-learn, mlflow, boto3, google-genai, langgraph, rapidfuzz, fastapi.
- `uv sync --group ml` adds torch, transformers and peft for fine-tuning.
- `uv sync --group deploy` adds aws-cdk-lib and constructs.

## 1. Secrets

- `GEMINI_API_KEY` (AI Studio key), only if you use `--provider gemini`. Export it in your shell;
  never commit it. Vertex AI and Bedrock need no key: they use your cloud identity.
- AWS credentials live in `~/.aws`. Pick the profile with `export AWS_PROFILE=<your-profile>`.
- Google credentials live in `~/.config/gcloud` (step 3c).

## 2. AWS (region us-east-1)

**a. Log in / refresh.** If a command says the token expired, renew it, then check with
`aws sts get-caller-identity`.

**b. CDK bootstrap (once per account and region).** It creates the `CDKToolkit` stack: an S3 bucket
for assets such as `model.tar.gz`, and the roles CDK deploys with. On a new account: `npx aws-cdk@latest bootstrap`.

**c. Use the latest CDK CLI.** The globally installed `cdk` (2.1100) is older than the library needs
(2.1144 or newer). Always call `npx aws-cdk@latest ...`. Optional:
`export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1` hides the Node version warning.

**d. SageMaker serverless memory quota.** New accounts allow **3,072 MB per serverless endpoint**.
The stack uses 3072 (the model is 332 MB). Going higher needs a Service Quotas request, which a
person at AWS approves, so it takes hours or days. Check:
```
aws service-quotas list-service-quotas --service-code sagemaker \
  --region us-east-1 --query "Quotas[?contains(QuotaName,'serverless')].[QuotaName,Value]" --output table
```

**e. Bedrock: Anthropic use-case form (once per account).** Console → Amazon Bedrock (us-east-1) →
Model catalog → a Claude model → submit the use-case details. It takes up to 15 minutes to activate.
Before that, calls fail with "Model use case details have not been submitted for this account".
On the account used here, Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) works. The
Claude 5.5 models are listed but not enabled ("not available for this account").

**f. Deploy and call.**
```
cd sectors/deploy/aws && npx aws-cdk@latest deploy
```
The outputs print `ApiUrl`, `ApiKeyId` and `EndpointName`. To call it:
```
KEY=$(aws apigateway get-api-key --api-key <ApiKeyId> \
  --include-value --query value --output text --region us-east-1)
curl -s -X POST <ApiUrl> -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"text": "<at least 50 characters of business description>"}'
```
The first call after an idle period is a cold start (24 s measured; API Gateway's limit is 29 s).
Warm calls take about 0.5 s.

## 3. Google Cloud (region us-central1)

**a. Project and billing.** Use the project *ID*, which is not always its display name.
`export GOOGLE_CLOUD_PROJECT=<your-project-id>`
- `gcloud config set project $GOOGLE_CLOUD_PROJECT`
- Link billing at https://console.cloud.google.com/billing/linkedaccount?project=$GOOGLE_CLOUD_PROJECT
- Check: `gcloud billing projects describe $GOOGLE_CLOUD_PROJECT --format="value(billingEnabled)"`

**b. APIs.** `deploy.sh` enables Cloud Run, Cloud Build and Artifact Registry. For Gemini on Vertex,
turn on the Agent Platform (Vertex AI, `aiplatform.googleapis.com`) in the console, or run
`gcloud services enable aiplatform.googleapis.com`.

**c. Two different logins.**
- `gcloud auth login`: for the `gcloud` command itself.
- `gcloud auth application-default login`: **Application Default Credentials (ADC)**, a file that
  Python libraries (google-genai with Vertex) read. Without it: "Your default credentials were not
  found".

**d. Gemini: API key or Vertex AI.**

| | Gemini API (AI Studio key) | Vertex AI (`--provider vertex`) |
|---|---|---|
| Auth | Shared API key in `.env` | Your IAM identity (ADC); a service account in production |
| Billing | **Prepaid credits** in AI Studio, separate from Cloud Billing | The project's Cloud Billing account |
| Free tier | A few requests a minute (429s); prompts may be used to improve Google products | None; data stays under the Cloud terms |
| Use for | Quick experiments | Anything with client data |

**e. The Cloud Run build account needs a role.** `gcloud run deploy --source` builds with Cloud Build,
which runs as the project's default compute service account. Newer projects don't grant it
build permissions. Grant the Cloud Run Builder role once:
```
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com \
  --role=roles/run.builder
```

**f. Deploy and call.** `bash sectors/deploy/gcp/deploy.sh`. On the first run, answer **Y** to
creating the Artifact Registry repository `cloud-run-source-deploy`. The service is private
(`--no-allow-unauthenticated`), so you call it with a Google identity token, not an API key:
```
RUN_URL=$(gcloud run services describe sector-classifier --region us-central1 --format='value(status.url)')
curl -s -X POST "$RUN_URL/classify" -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" -d '{"text": "<at least 50 characters>"}'
```

## 4. Tracking

MLflow writes to `mlflow.db` (experiment "sectors"); nothing to set up. Browse it with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. Every script's console output is
also saved to `out/logs/<script>_<date>_<time>.txt`.

## 5. Errors we hit and the fix

| Where | Error | Cause | Fix |
|---|---|---|---|
| `cdk deploy` | CLI too old for the library | Global `cdk` 2.1100 | `npx aws-cdk@latest` |
| `cdk synth` | Stack creates an account-wide `ApiGateway::Account` | CDK's default CloudWatch role for API Gateway | `cloud_watch_role=False` |
| `cdk deploy` | `ServiceLimitExceeded`: memory per serverless endpoint is 3072 MB | Account quota | 3072 MB in `sector_stack.py` |
| Redeploy after a failed first create | Stack stuck in `ROLLBACK_IN_PROGRESS` | CloudFormation deletes what it made | Wait for `ROLLBACK_COMPLETE`; `cdk deploy` then deletes and recreates the stack itself |
| `finetune.py` | `_Tee` has no attribute `isatty` | The log tee hid the screen's attributes | `__getattr__` passes them through |
| `finetune.py` | Unexpected keyword `warmup_ratio` | transformers 5.20 renamed it | `warmup_steps=0.1` |
| `llm_classify.py` gemini | 429 RESOURCE_EXHAUSTED | Free tier: billing not linked | Link billing, or use Vertex |
| `llm_classify.py` gemini | 402 prepayment credits depleted | The AI Studio key bills from prepaid credits | Use `--provider vertex`, or buy AI Studio credits |
| `llm_classify.py` vertex | Default credentials not found | No ADC file | `gcloud auth application-default login` |
| `llm_classify.py` bedrock | Model use case details not submitted | Anthropic form not filled in | Step 2e, then wait 15 minutes |
| `deploy.sh` | PERMISSION_DENIED for `...-compute@developer.gserviceaccount.com` | The build account lacks roles | Step 3e |

The script fixes that came out of these errors: `llm_classify.py` retries only errors that can clear
up (429 and 5xx), stops after 10 failures in a row, and won't log a run to MLflow if more than 5% of
its calls failed.

## 6. Costs while things are up

| Piece | Idle | Per use |
|---|---|---|
| SageMaker serverless, Lambda, API Gateway | $0 | Per request and per millisecond of compute |
| Cloud Run (`--min-instances 0`) | $0 | Per request and per second of CPU and memory |
| S3 and Artifact Registry (model and image) | Cents a month for storage | n/a |
| Zero-shot LLM runs (853 filings) | n/a | About $1.40 (Gemini 3.8 Flash), $2.00 (Claude Haiku 4.5) |

Tear everything down with `sectors/deploy/TEARDOWN.md` when you are done.
