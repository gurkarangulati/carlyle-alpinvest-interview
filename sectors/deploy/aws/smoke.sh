#!/usr/bin/env bash
# Call the deployed AWS API: twice (cold, then warm), then once without a key.
# Reads the URL and key from the stack's outputs, so nothing is pasted by hand.
# Run from the repo root:  AWS_PROFILE=<profile> bash sectors/deploy/aws/smoke.sh
set -euo pipefail
: "${AWS_PROFILE:?set AWS_PROFILE to your AWS CLI profile}"
REGION=${AWS_REGION:-us-east-1}

output() {
  aws cloudformation describe-stacks --stack-name SectorClassifier --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
URL=$(output ApiUrl)
KEY=$(aws apigateway get-api-key --api-key "$(output ApiKeyId)" --include-value \
  --query value --output text --region "$REGION")   # held in a variable, never printed
BODY='{"text": "We explore for, develop and produce crude oil and natural gas from our wells in the Permian Basin and operate pipelines that carry it to refineries."}'

for call in first second; do
  echo "$call call:"
  curl -s -w '\n  HTTP %{http_code} in %{time_total}s\n' -X POST "$URL" \
    -H "x-api-key: $KEY" -H "Content-Type: application/json" -d "$BODY"
done
echo "no API key (expect 403):"
curl -s -o /dev/null -w '  HTTP %{http_code}\n' -X POST "$URL" -H "Content-Type: application/json" -d "$BODY"
