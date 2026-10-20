"""Ask the DEPLOYED agent a question (AgentCore Runtime in AWS).

    AWS_PROFILE=<profile> uv run python agent/invoke.py --question "..."
    (or: make ask-aws Q="...")

Nothing is pasted by hand: the runtime ARN is read from the AnalystAgent stack's
CloudFormation outputs. The call is boto3 `bedrock-agentcore` invoke_agent_runtime;
the payload is the same {"prompt": ...} that app.py receives locally.
"""

import argparse
import json
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import os

import boto3
from botocore.config import Config

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STACK = "AnalystAgent"
REGION = os.environ.get("AWS_REGION", "us-east-1")


@lru_cache(maxsize=1)
def runtime_arn() -> str:
    cfn = boto3.client("cloudformation", region_name=REGION)
    outputs = cfn.describe_stacks(StackName=STACK)["Stacks"][0].get("Outputs", [])
    for o in outputs:
        if o["OutputKey"] == "RuntimeArn":
            return o["OutputValue"]
    raise RuntimeError(f"stack {STACK} has no RuntimeArn output; was it deployed?")


@lru_cache(maxsize=1)
def _client():
    # The agent may take ~30 s (several Bedrock turns + Athena queries).
    return boto3.client("bedrock-agentcore", region_name=REGION,
                        config=Config(read_timeout=180, retries={"mode": "standard"}))


def invoke(question: str, session_id: str | None = None) -> dict:
    """One question to the deployed runtime; returns app.py's JSON result."""
    r = _client().invoke_agent_runtime(
        agentRuntimeArn=runtime_arn(),
        qualifier="DEFAULT",
        # A session id must be at least 33 characters; a new one = a fresh microVM session.
        runtimeSessionId=session_id or f"analyst-{uuid.uuid4().hex}",
        contentType="application/json",
        accept="application/json",
        payload=json.dumps({"prompt": question}).encode())
    body = r["response"].read() if hasattr(r["response"], "read") else r["response"]
    return json.loads(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--trace", action="store_true", help="print the full trace too")
    args = parser.parse_args()
    out = invoke(args.question)
    print(f"answer     {out.get('answer')}")
    print(f"value      {out.get('answer_value')!r}")
    print(f"citations  {out.get('citations')}")
    print(f"status     {out.get('status')}  ({out.get('reason')})")
    print(f"tool calls {out.get('tool_calls')}   cost ${out.get('cost_usd') or 0:.4f}   "
          f"{out.get('seconds')}s in the runtime")
    if args.trace:
        print(json.dumps(out.get("trace"), indent=2))
