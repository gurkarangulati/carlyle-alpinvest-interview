#!/usr/bin/env bash
# Call the Cloud Run service: twice with your Google identity, then once without (expect 403).
# Run from the repo root:  GOOGLE_CLOUD_PROJECT=<project-id> bash sectors/deploy/gcp/smoke.sh
set -euo pipefail
PROJECT=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}
: "${PROJECT:?set GOOGLE_CLOUD_PROJECT}"
URL=$(gcloud run services describe sector-classifier --region us-central1 --project "$PROJECT" \
  --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)   # short-lived, tied to who you are
BODY='{"text": "We explore for, develop and produce crude oil and natural gas from our wells in the Permian Basin and operate pipelines that carry it to refineries."}'

for call in first second; do
  echo "$call call:"
  curl -s -w '\n  HTTP %{http_code} in %{time_total}s\n' -X POST "$URL/classify" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "$BODY"
done
echo "no identity token (expect 403):"
curl -s -o /dev/null -w '  HTTP %{http_code}\n' -X POST "$URL/classify" -H "Content-Type: application/json" -d "$BODY"
