"""The inference service. The same file runs on the Mac and inside the Cloud Run container.

Local:  cd sectors/deploy/app && MODEL_DIR=../../out/models/finetune_full \
            uv run --group deploy uvicorn main:app --port 8080
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.8"))  # config, not code: easy to change
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, never per request.
    state["clf"] = pipeline("text-classification", model=MODEL_DIR, device="cpu")
    yield


app = FastAPI(title="Sector classifier", lifespan=lifespan)


class Filing(BaseModel):
    text: str = Field(min_length=50, max_length=100_000)  # reject junk before the model


@app.get("/health")
def health():
    return {"ok": "clf" in state}


@app.post("/classify")
def classify(filing: Filing):
    top = state["clf"](filing.text, truncation=True, max_length=256)[0]
    route = "auto_tag" if top["score"] >= THRESHOLD else "analyst_review"
    return {"sector": top["label"], "confidence": round(top["score"], 3), "route": route}
