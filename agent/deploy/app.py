"""CDK entry point: `cdk synth|deploy|destroy` runs this file to learn what to build.

Two stacks, so the lake can live without the agent:
    AnalystLake    S3 + Glue + Athena          (make deploy-lake)
    AnalystAgent   AgentCore Runtime + role    (make deploy-agent; needs AnalystLake)
"""

import os
import sys
from pathlib import Path

import aws_cdk as cdk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root: for agent.schema

from agent_stack import AgentStack  # noqa: E402
from lake_stack import LakeStack  # noqa: E402

app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region="us-east-1")
lake = LakeStack(app, "AnalystLake", env=env)
AgentStack(app, "AnalystAgent", lake=lake, env=env)
for stack in app.node.children:
    if isinstance(stack, cdk.Stack):
        cdk.Tags.of(stack).add("project", "analyst-agent")
app.synth()
