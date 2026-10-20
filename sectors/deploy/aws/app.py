"""CDK entry point: `cdk deploy` runs this file to learn what to build."""

import os

import aws_cdk as cdk

from sector_stack import SectorStack

app = cdk.App()
SectorStack(app, "SectorClassifier",
            env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                                region=os.environ.get("AWS_REGION", "us-east-1")))
app.synth()
