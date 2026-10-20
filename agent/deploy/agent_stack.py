"""AgentStack: the agent container on Amazon Bedrock AgentCore Runtime.

    DockerImageAsset (linux/arm64)  ->  ECR (CDK bootstrap repo)
    AgentCore Runtime               runs the image in per-session microVMs, port 8080
    IAM execution role              what the agent's code may do in AWS, and nothing else

The runtime gets the lake's names as environment variables, so the same image runs
against any lake: LAKE_BACKEND=athena, ATHENA_DATABASE, ATHENA_WORKGROUP, ATHENA_OUTPUT.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from constructs import Construct

from lake_stack import LAKE_PREFIX, RESULTS_PREFIX, LakeStack

ML = Path(__file__).resolve().parents[2]          # the repo root: the Docker build context
MODEL_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
# The "us." cross-region inference profile may route a call to any of these regions,
# so the role needs the foundation model in each of them.
PROFILE_REGIONS = ["us-east-1", "us-east-2", "us-west-2"]
RUNTIME_NAME = "analyst_agent"

# Same list as agent/Dockerfile.dockerignore: CDK hashes and uploads only these files.
DOCKER_CONTEXT = [
    "*",
    "!agent/*.py", "!agent/requirements.txt", "!agent/Dockerfile",
    "!matching/__init__.py", "!matching/matcher.py", "!matching/results/hybrid_thresholds.json",
    "!extraction/llm.py", "!extraction/qa_tools.py", "!extraction/qa_agent.py",
    "!extraction/common.py",
]


class AgentStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, *, lake: LakeStack, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. The image: built for ARM64 (AgentCore runs Graviton), pushed by `cdk deploy`.
        image = ecr_assets.DockerImageAsset(
            self, "AgentImage",
            directory=str(ML),
            file="agent/Dockerfile",
            platform=ecr_assets.Platform.LINUX_ARM64,
            exclude=DOCKER_CONTEXT,
            ignore_mode=cdk.IgnoreMode.DOCKER)

        # 2. The execution role. AgentCore assumes it to run our container; only calls
        #    from this account's AgentCore runtimes may assume it (confused-deputy guard).
        role = iam.Role(
            self, "AgentRole",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {"aws:SourceArn":
                                f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:*"}}),
            description="Analyst agent: Bedrock Haiku, Athena on one workgroup, read the lake")

        bucket = lake.bucket
        statements = [
            # Bedrock: only Claude Haiku 4.5, via the US inference profile.
            iam.PolicyStatement(
                sid="InvokeHaiku",
                actions=["bedrock:InvokeModel"],
                resources=[f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/"
                           f"{MODEL_PROFILE}"] +
                          [f"arn:aws:bedrock:{r}::foundation-model/{MODEL}" for r in PROFILE_REGIONS]),
            # Athena: run and read queries in our workgroup only.
            iam.PolicyStatement(
                sid="AthenaWorkgroup",
                actions=["athena:StartQueryExecution", "athena:GetQueryExecution",
                         "athena:GetQueryResults", "athena:StopQueryExecution",
                         "athena:GetWorkGroup"],
                resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/"
                           f"{lake.workgroup_name}"]),
            # Glue: read the catalog entries of our database (Athena checks them as the caller).
            iam.PolicyStatement(
                sid="GlueRead",
                actions=["glue:GetDatabase", "glue:GetTable", "glue:GetTables",
                         "glue:GetPartition", "glue:GetPartitions"],
                resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog",
                           f"arn:aws:glue:{self.region}:{self.account}:database/{lake.database_name}",
                           f"arn:aws:glue:{self.region}:{self.account}:table/{lake.database_name}/*"]),
            # S3: read the lake prefix; read and write the results prefix; nothing else.
            iam.PolicyStatement(
                sid="LakeRead",
                actions=["s3:GetObject"],
                resources=[bucket.arn_for_objects(f"{LAKE_PREFIX}/*")]),
            iam.PolicyStatement(
                sid="ResultsReadWrite",
                actions=["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload",
                         "s3:ListMultipartUploadParts"],
                resources=[bucket.arn_for_objects(f"{RESULTS_PREFIX}/*")]),
            iam.PolicyStatement(
                sid="BucketList",
                actions=["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"],
                resources=[bucket.bucket_arn]),
        ]
        for s in statements:
            role.add_to_policy(s)
        # Pull our image (ECR GetAuthorizationToken has no resource-level scoping).
        image.repository.grant_pull(role)
        # The L2 Runtime below adds the rest itself (visible in cdk.out/AnalystAgent.template.json):
        # CloudWatch Logs on /aws/bedrock-agentcore/runtimes/*, X-Ray, cloudwatch:PutMetricData,
        # and bedrock-agentcore:GetWorkloadAccessToken* on the default workload identity.

        # 3. The runtime. L2 construct; it wires the image and the role together.
        runtime = agentcore.Runtime(
            self, "Runtime",
            runtime_name=RUNTIME_NAME,
            description="LangGraph analyst agent over the S3/Glue/Athena lake",
            agent_runtime_artifact=agentcore.AgentRuntimeArtifact.from_ecr_repository(
                image.repository, image.image_tag),
            execution_role=role,
            protocol_configuration=agentcore.ProtocolType.HTTP,
            network_configuration=agentcore.RuntimeNetworkConfiguration.using_public_network(),
            lifecycle_configuration=agentcore.LifecycleConfiguration(
                idle_runtime_session_timeout=cdk.Duration.minutes(5),
                max_lifetime=cdk.Duration.hours(1)),
            environment_variables={
                "LAKE_BACKEND": "athena",
                "ATHENA_DATABASE": lake.database_name,
                "ATHENA_WORKGROUP": lake.workgroup_name,
                "ATHENA_OUTPUT": lake.results_location,
            })
        runtime.node.add_dependency(role)  # permissions exist before the runtime starts

        cdk.CfnOutput(self, "RuntimeArn", value=runtime.agent_runtime_arn)
        cdk.CfnOutput(self, "RuntimeId", value=runtime.agent_runtime_id)
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name)
        cdk.CfnOutput(self, "GlueDatabase", value=lake.database_name)
        cdk.CfnOutput(self, "AthenaWorkgroup", value=lake.workgroup_name)
