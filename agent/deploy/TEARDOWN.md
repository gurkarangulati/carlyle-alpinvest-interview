# Teardown: the lake and the agent on AWS

Read this before `make deploy-all`. Everything here runs in `us-east-1` with the AWS
profile you pass as `PROFILE=` (or `AWS_PROFILE`).

Two CloudFormation stacks, destroyed in this order (the agent stack imports names
from the lake stack, so the lake stack cannot go first):

1. `AnalystAgent`: the AgentCore Runtime, its IAM role, the container image reference.
2. `AnalystLake`: the S3 bucket, the Glue database and tables, the Athena workgroup.

```bash
make teardown-all PROFILE=<profile>        # = teardown-agent, then teardown-lake
```

## What `cdk destroy` removes

| Stack | Resource | Removed? |
|---|---|---|
| AnalystAgent | AgentCore Runtime (and its DEFAULT endpoint) | yes |
| AnalystAgent | IAM execution role and its policies | yes |
| AnalystLake | S3 bucket, **including every object**: `RemovalPolicy.DESTROY` + `auto_delete_objects=True` adds a small Lambda that empties the bucket before CloudFormation deletes it (lake Parquet and Athena results) | yes |
| AnalystLake | Glue database and the 8 tables (metadata only; the data lived in the bucket) | yes |
| AnalystLake | Athena workgroup (`recursive_delete_option` deletes its saved queries and history with it) | yes |
| AnalystLake | The helper Lambdas for BucketDeployment and auto-delete, and their roles | yes |

## What it leaves behind (and why)

| Left behind | Why | Cost while it sits there |
|---|---|---|
| The agent's container image in the CDK bootstrap ECR repo `cdk-hnb659fds-container-assets-<account>-us-east-1` | CDK assets belong to the bootstrap stack, shared by every CDK app in the account, so `cdk destroy` never deletes them | ECR storage, ~$0.10 per GB-month (the image is a few hundred MB) |
| The Parquet and Docker-context zip files in the CDK bootstrap S3 bucket `cdk-hnb659fds-assets-<account>-us-east-1` | Same reason: bootstrap assets | S3 storage, cents |
| CloudWatch log groups: `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT` (the agent's logs) and `/aws/lambda/AnalystLake-*` (the helper Lambdas) | Log groups created by a service at run time are not part of the stack | $0.03 per GB-month stored |
| The CDK bootstrap stack `CDKToolkit` itself | Shared by all CDK apps; keep it if you use CDK again | ~nothing while unused |

## Cleanup commands (run after `cdk destroy`)

`make teardown-agent` and `make teardown-lake` run these for you. By hand:

```bash
export AWS_PROFILE=<profile> AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# 1. Log groups the stacks did not own (the agent's runtime logs, the helper Lambdas).
for g in $(aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/analyst_agent \
             --query 'logGroups[].logGroupName' --output text); do aws logs delete-log-group --log-group-name "$g"; done
for g in $(aws logs describe-log-groups --log-group-name-prefix /aws/lambda/AnalystLake \
             --query 'logGroups[].logGroupName' --output text); do aws logs delete-log-group --log-group-name "$g"; done

# 2. CDK asset images. This deletes EVERY image in the bootstrap repo, so only do it
#    if no other CDK app in this account still needs its image (or re-deploy those after).
REPO=cdk-hnb659fds-container-assets-$ACCOUNT-us-east-1
IDS=$(aws ecr list-images --repository-name $REPO --query 'imageIds[*]' --output json)
[ "$IDS" != "[]" ] && aws ecr batch-delete-image --repository-name $REPO --image-ids "$IDS"

# 3. (Optional) remove CDK bootstrap entirely, if you will not use CDK in this account again.
#    aws cloudformation delete-stack --stack-name CDKToolkit   (empty its S3 bucket and ECR repo first)
```

## Check that nothing is left

```bash
aws cloudformation describe-stacks --query "Stacks[?starts_with(StackName,'Analyst')].StackName"   # []
aws bedrock-agentcore-control list-agent-runtimes --query 'agentRuntimes[].agentRuntimeName'       # no analyst_agent
aws athena list-work-groups --query "WorkGroups[?Name=='analyst-agent']"                           # []
aws glue get-database --name analyst_lake 2>&1 | grep -q EntityNotFound && echo "glue: gone"
aws s3 ls | grep -i analystlake || echo "s3: gone"
```

## If a destroy fails

- **AnalystLake fails with "Export ... is in use"**: the agent stack still exists. Destroy
  `AnalystAgent` first.
- **Bucket not empty**: the auto-delete Lambda was removed before it ran (e.g. a stack
  that was edited by hand). Empty it with `aws s3 rm s3://<bucket> --recursive`, then retry.
- **Workgroup not empty**: only if `recursive_delete_option` was turned off; delete it with
  `aws athena delete-work-group --work-group analyst-agent --recursive-delete-option`.
