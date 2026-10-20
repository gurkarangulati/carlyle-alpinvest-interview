"""HTTP API for the matcher.

Run:   uv run uvicorn matching.api:app --port 8000 --no-access-log
Try:   curl "http://127.0.0.1:8000/match?name=Walt%20Disney%20Co&k=3"
       curl "http://127.0.0.1:8000/match?name=Walt%20Disney%20Co&as_of=2015-06-30"
       curl -X POST http://127.0.0.1:8000/mentions -H "Content-Type: application/json" \
            -d '{"text": "We compete with Acme Widgets, Inc. and Apple Inc."}'
       curl http://127.0.0.1:8000/health
Docs:  http://127.0.0.1:8000/docs

--no-access-log matters: the access log would write every query string (the
names people look up) to the console. This service never logs request text.
"""

from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from matching.matcher import load_matcher
from matching.mentions import MAX_TEXT, link_mentions

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index once at startup (a few seconds), not on every request.
    state["matcher"] = load_matcher()
    yield
    state.clear()


app = FastAPI(title="Company name matcher", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Say what was wrong, but do not echo the input back (it may be confidential)."""
    errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class Candidate(BaseModel):
    cik: int
    name: str  # current name
    alias: str | None  # the name or ticker that matched
    alias_kind: str  # current, former or ticker
    score: float  # 0-100


class MatchResponse(BaseModel):
    decision: str  # "match" or "abstain"
    candidates: list[Candidate]


class MentionsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    as_of: date | None = None


class Mention(BaseModel):
    mention: str
    start: int
    end: int
    decision: str
    match: Candidate | None


@app.get("/health")
def health():
    m = state.get("matcher")
    return {"status": "ok" if m else "loading",
            "companies": len(m.names) if m else 0, "aliases": len(m.aliases) if m else 0}


@app.get("/match", response_model=MatchResponse)
def match(name: str = Query(min_length=1, max_length=200),
          k: int = Query(5, ge=1, le=20),
          as_of: date | None = None):
    """Ranked candidates for one name, plus match or abstain."""
    r = state["matcher"].match(name, k, as_of.isoformat() if as_of else None)
    return {"decision": r["decision"], "candidates": r["candidates"]}


@app.post("/mentions", response_model=list[Mention])
def mentions(body: MentionsRequest):
    """Every company name found in the text, each with its match (or abstain)."""
    as_of = body.as_of.isoformat() if body.as_of else None
    return link_mentions(state["matcher"], body.text, as_of)
