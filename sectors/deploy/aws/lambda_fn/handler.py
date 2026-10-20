"""API Gateway -> this Lambda -> SageMaker endpoint.

Validates the request, calls the model, and applies the business rule:
confident answers are auto-tagged, the rest go to an analyst.
"""

import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")  # created once per container, reused per call
ENDPOINT = os.environ["ENDPOINT_NAME"]
THRESHOLD = float(os.environ["THRESHOLD"])
MAX_CHARS = 2000  # the model reads ~1,000 characters (256 tokens); don't ship more


def reply(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body)}


def handler(event, context):
    try:
        text = json.loads(event.get("body") or "{}")["text"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return reply(400, {"error": 'send JSON like {"text": "..."}'})
    if not isinstance(text, str) or len(text) < 50:
        return reply(400, {"error": "text must be at least 50 characters"})

    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"inputs": text[:MAX_CHARS],
                         "parameters": {"truncation": True, "max_length": 256}}),
    )
    top = json.loads(resp["Body"].read())[0]  # [{"label": "energy", "score": 0.97}]
    route = "auto_tag" if top["score"] >= THRESHOLD else "analyst_review"
    print(json.dumps({"chars": len(text), "sector": top["label"], "route": route}))  # never log the text
    return reply(200, {"sector": top["label"], "confidence": round(top["score"], 3),
                       "route": route})
