# Security

How this repository keeps secrets and account details out of the code, how the deployed pieces
are locked down, and how that is checked.

## Nothing account-specific is committed

| What | How it's supplied |
|---|---|
| AWS account | Read at deploy time from your credentials (`CDK_DEFAULT_ACCOUNT`); every IAM ARN is built from the deploying account and region |
| AWS profile and region | `AWS_PROFILE` and `AWS_REGION` (`make … PROFILE=<profile> REGION=<region>`); default region us-east-1 |
| Google Cloud project and region | `GOOGLE_CLOUD_PROJECT` and `GCP_REGION` (`make … PROJECT=<id> GCP_REGION=<region>`) |
| Gemini API key (only if not using Vertex AI) | `GEMINI_API_KEY` in your shell; Vertex AI and Bedrock use cloud identity, no keys |
| SEC contact (required by the SEC's fair-access policy) | `SEC_USER_AGENT="Your Name you@example.com"` |
| Resource names and the auto-tag threshold | CDK context with defaults: `-c lake_database=… -c workgroup=… -c runtime_name=… -c threshold=…` |
| The classifier's API key value | Never stored: the stack outputs only the key's ID; `smoke.sh` fetches the value at run time into a variable it never prints |

**Never committed, by `.gitignore`:**
- `.env`;
- `data/` (all of it is rebuilt by `make`);
- model files;
- `mlflow.db` and `mlruns/`;
- `cdk.out/`.

The one 12-digit account number in the code, `763104351884`, is **AWS's public Deep Learning
Containers registry** that hosts the Hugging Face image, not an account of ours.

## How the deployed pieces are locked down

- **Least-privilege IAM.**
  - The agent's role may invoke one Bedrock model, run queries in one Athena workgroup, read
    one Glue database, and read the lake. It may write only Athena results.
  - The classifier's Lambda may invoke one SageMaker endpoint.
  - Trust policies are scoped to this account (`aws:SourceAccount`, `aws:SourceArn`).
- **Storage.** The S3 bucket blocks all public access, encrypts at rest, denies any request not
  using TLS, and empties itself on teardown.
- **Containers.** Both run as a non-root user.
- **No write path for the agent.**
  - Its SQL is parsed (sqlglot) and limited to a single SELECT over allowlisted tables, with a
    row limit; 61 attack cases are tested.
  - The engine is read-only, with a 1 GB scan cutoff per query.
- **Services.**
  - Cloud Run is private (IAM identity token required).
  - The AWS API requires a key and is throttled. Production would use IAM, Cognito or a Lambda
    authorizer (see `docs/ARCHITECTURE.md`).
- **Logs** hold request metadata and traces, never the documents' full text.

## How it is checked

| Check | Command | Also runs |
|---|---|---|
| Secrets in the files and the full git history (gitleaks) | `make security-scan` | On every push (`.github/workflows/security.yml`) |
| Known vulnerabilities in every locked and container dependency (pip-audit) | `make deps-audit` | On every push |
| Infrastructure changes that widen permissions | `cdk deploy --require-approval broadening` (built into `make deploy-*`) | Every deploy |
| SQL guard, verification and input validation | `make test` | — |

## Known limits (demo scope)

- API-key auth on the classifier, rather than IAM or Cognito.
- No WAF, no VPC endpoints, no customer-managed KMS keys.
- No Bedrock Guardrails on the agents yet; the documents they read are treated as data, but a
  prompt-attack filter is the recommended next step (`docs/IMPROVEMENTS.md`).

## Reporting

This is a personal sample project. If you notice anything sensitive, please open an issue
without including the sensitive value itself.
