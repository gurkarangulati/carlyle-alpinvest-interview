# Teardown (written before anything was created)

## AWS (us-east-1; set AWS_PROFILE to your profile first)
1. `cd sectors/deploy/aws && npx aws-cdk@latest destroy`
   Deletes API Gateway, Lambda, the SageMaker endpoint, endpoint config, model and IAM roles.
2. Log groups outlive the stack. Delete them:
   `aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints/`
   `aws logs describe-log-groups --log-group-name-prefix /aws/lambda/SectorClassifier`
   then `aws logs delete-log-group --log-group-name <name>`
3. The model tarball stays in the CDK assets bucket (cdk-hnb659fds-assets-<account>-us-east-1).
   Delete the large .tar.gz object there.
4. Check nothing is left: `aws sagemaker list-endpoints`

## Google Cloud (project <your-project-id>, us-central1)
1. `gcloud run services delete sector-classifier --region us-central1`
2. Delete the image: Artifact Registry repo `cloud-run-source-deploy` in us-central1
   (`gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1`).
3. The uploaded source zips: `gs://run-sources-<your-project-id>-us-central1`
   (`gcloud storage rm --recursive gs://run-sources-<your-project-id>-us-central1`).
4. Optional: remove the build role granted in ONBOARDING step 3e
   (`gcloud projects remove-iam-policy-binding <your-project-id> --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/run.builder`).
