"""The AWS architecture as code: API Gateway -> Lambda -> SageMaker serverless endpoint."""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3_assets as s3_assets
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

TARBALL = Path(__file__).resolve().parents[2] / "out" / "model.tar.gz"
# AWS's prebuilt Hugging Face inference container (CPU). It loads model.tar.gz
# and serves it; HF_TASK tells it which pipeline to build. No inference code needed.
HF_IMAGE = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:"
            "2.6.0-transformers5.5.3-cpu-py312-ubuntu22.04")


class SectorStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1-2. The model artifact. CDK uploads the tarball to its S3 bucket.
        model_data = s3_assets.Asset(self, "ModelData", path=str(TARBALL))

        # 3. The role SageMaker assumes: read this one file, pull the image, write logs.
        role = iam.Role(self, "SageMakerRole",
                        assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        model_data.grant_read(role)
        role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                     "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
            resources=["*"]))
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                     "cloudwatch:PutMetricData"],
            resources=["*"]))

        # 4. Model: which container runs which weights.
        model = sagemaker.CfnModel(
            self, "Model",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=HF_IMAGE,
                model_data_url=model_data.s3_object_url,
                environment={"HF_TASK": "text-classification"}))
        model.node.add_dependency(role)  # permissions exist before SageMaker uses them

        # 5. Endpoint config: HOW it runs. Serverless = scales to zero, pay per request.
        # 3072 MB = this account's default quota per serverless endpoint (model is 332 MB).
        config = sagemaker.CfnEndpointConfig(
            self, "EndpointConfig",
            production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                model_name=model.attr_model_name,
                variant_name="AllTraffic",
                serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                    memory_size_in_mb=3072, max_concurrency=2))])

        # 6. The live endpoint.
        endpoint = sagemaker.CfnEndpoint(
            self, "Endpoint", endpoint_config_name=config.attr_endpoint_config_name)

        # 7. Lambda: validate the request, call the endpoint, apply the business rule.
        fn = _lambda.Function(
            self, "ClassifyFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=_lambda.Code.from_asset(str(Path(__file__).parent / "lambda_fn")),
            timeout=cdk.Duration.seconds(60),
            memory_size=256,
            environment={"ENDPOINT_NAME": endpoint.attr_endpoint_name, "THRESHOLD": "0.8"})
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"], resources=[endpoint.ref]))  # ref = its ARN

        # 8. API Gateway: the HTTPS front door, API key required, throttled.
        api = apigw.LambdaRestApi(
            self, "Api", handler=fn, proxy=False,
            cloud_watch_role=False,  # don't change the account-wide API Gateway setting
            deploy_options=apigw.StageOptions(throttling_rate_limit=5,
                                              throttling_burst_limit=10))
        api.root.add_resource("classify").add_method("POST", api_key_required=True)
        plan = api.add_usage_plan(
            "Plan", throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10),
            quota=apigw.QuotaSettings(limit=1000, period=apigw.Period.DAY))
        key = api.add_api_key("Key")
        plan.add_api_key(key)
        plan.add_api_stage(stage=api.deployment_stage)

        # What `cdk deploy` prints at the end.
        cdk.CfnOutput(self, "ApiUrl", value=f"{api.url}classify")
        cdk.CfnOutput(self, "ApiKeyId", value=key.key_id)
        cdk.CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
