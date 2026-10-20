#!/usr/bin/env bash
# Google Cloud: build the container from source with Cloud Build, run it on Cloud Run.
# Run from the repo root:  GOOGLE_CLOUD_PROJECT=<project-id> bash sectors/deploy/gcp/deploy.sh
set -euo pipefail
PROJECT=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}
: "${PROJECT:?set GOOGLE_CLOUD_PROJECT}"
cd "$(dirname "$0")/../app"
rm -rf model && cp -R ../../out/models/finetune_full model   # bake the model into the image

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  --project "$PROJECT"
gcloud run deploy sector-classifier \
  --project "$PROJECT" \
  --source . \
  --region us-central1 \
  --memory 4Gi --cpu 2 \
  --min-instances 0 --max-instances 2 \
  --no-allow-unauthenticated
