"""LakeStack: the data lake on AWS. S3 holds the Parquet, Glue describes it, Athena queries it.

    s3://<bucket>/lake/silver/<table>/data.parquet     cleaned facts (5 tables)
    s3://<bucket>/lake/gold/<table>/data.parquet       analysis-ready (3 tables)
    s3://<bucket>/athena-results/                      query result files (Athena writes)

Glue tables are written out column by column from agent/schema.py (the same contract the
agent's describe_tables shows the model), not discovered by a crawler: a crawler guesses
types and can change them on the next run; an explicit schema fails loudly instead.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_athena as athena
from aws_cdk import aws_glue as glue
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

from agent.schema import TABLES

LAKE_DIR = Path(__file__).resolve().parents[2] / "data" / "lake"
DATABASE = "analyst_lake"
WORKGROUP = "analyst-agent"
LAKE_PREFIX = "lake"
RESULTS_PREFIX = "athena-results"
SCAN_CUTOFF_BYTES = 1 * 1024**3   # Athena cancels any query that would scan more than 1 GB

PARQUET = {
    "input_format": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
    "output_format": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
    "serde": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
}


class LakeStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. The bucket. DESTROY + auto_delete_objects: `cdk destroy` empties and deletes it
        #    (fine for a rebuildable lake; a real lake would RETAIN and version).
        self.bucket = s3.Bucket(
            self, "LakeBucket",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(prefix=f"{RESULTS_PREFIX}/",
                                              expiration=cdk.Duration.days(7))])

        # 2. Upload the Parquet files (silver and gold only; bronze stays local).
        for layer in ("silver", "gold"):
            source = LAKE_DIR / layer
            if not source.exists():
                cdk.Annotations.of(self).add_warning_v2(
                    f"lake:{layer}", f"{source} not found: build the lake before deploy-lake")
                continue
            s3deploy.BucketDeployment(
                self, f"Upload{layer.title()}",
                sources=[s3deploy.Source.asset(str(source),
                                               exclude=["**/.DS_Store", "**/*.tmp"])],
                destination_bucket=self.bucket,
                destination_key_prefix=f"{LAKE_PREFIX}/{layer}/",
                memory_limit=512)

        # 3. The Glue catalog: one database, one table per contract table.
        self.database_name = DATABASE
        db = glue.CfnDatabase(
            self, "Database", catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=DATABASE, description="Analyst agent lake (silver + gold Parquet)"))
        self.table_names = []
        for table, (layer, purpose, cols) in TABLES.items():
            t = glue.CfnTable(
                self, f"Table{''.join(w.title() for w in table.split('_'))}",
                catalog_id=self.account,
                database_name=DATABASE,
                table_input=glue.CfnTable.TableInputProperty(
                    name=table,
                    description=purpose,
                    table_type="EXTERNAL_TABLE",
                    parameters={"classification": "parquet", "EXTERNAL": "TRUE"},
                    storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                        location=f"s3://{self.bucket.bucket_name}/{LAKE_PREFIX}/{layer}/{table}/",
                        input_format=PARQUET["input_format"],
                        output_format=PARQUET["output_format"],
                        serde_info=glue.CfnTable.SerdeInfoProperty(
                            serialization_library=PARQUET["serde"]),
                        columns=[glue.CfnTable.ColumnProperty(name=c, type=ty, comment=d[:255])
                                 for c, ty, d in cols])))
            t.node.add_dependency(db)
            self.table_names.append(table)

        # 4. The Athena workgroup: results in our bucket, a per-query scan cap, and the
        #    settings enforced so a client cannot point results elsewhere or lift the cap.
        self.results_location = f"s3://{self.bucket.bucket_name}/{RESULTS_PREFIX}/"
        self.workgroup_name = WORKGROUP
        athena.CfnWorkGroup(
            self, "WorkGroup",
            name=WORKGROUP,
            description="Analyst agent queries",
            recursive_delete_option=True,   # destroy also deletes its query history
            state="ENABLED",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                bytes_scanned_cutoff_per_query=SCAN_CUTOFF_BYTES,
                enforce_work_group_configuration=True,
                publish_cloud_watch_metrics_enabled=True,
                engine_version=athena.CfnWorkGroup.EngineVersionProperty(
                    selected_engine_version="Athena engine version 3"),
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=self.results_location,
                    encryption_configuration=athena.CfnWorkGroup.EncryptionConfigurationProperty(
                        encryption_option="SSE_S3"))))

        cdk.CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        cdk.CfnOutput(self, "GlueDatabase", value=DATABASE)
        cdk.CfnOutput(self, "AthenaWorkgroup", value=WORKGROUP)
        cdk.CfnOutput(self, "AthenaOutput", value=self.results_location)
