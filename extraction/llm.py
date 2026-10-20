"""One place that talks to the LLM: Amazon Bedrock, Claude Haiku 4.5, via `converse`.

AWS credentials come from the environment (AWS_PROFILE or the usual AWS env vars);
nothing about the account is written here.

What this file adds around the raw API call:
  - retry only errors that clear up on their own (throttling / 429 and 5xx),
    with exponential backoff; a 4xx like "access denied" fails at once;
  - a circuit breaker: after 10 failed calls in a row, stop calling;
  - token counts on every call, so the scripts can report cost.
"""

import random
import threading
import time

import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("AWS_REGION", "us-east-1")
PRICE_IN, PRICE_OUT = 1.00, 5.00  # USD per 1M tokens, Haiku 4.5 list price

TRANSIENT_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
                   "InternalServerException", "ModelNotReadyException", "ModelErrorException"}
MAX_ATTEMPTS = 4
MAX_FAILURES_IN_A_ROW = 10


class CircuitOpen(RuntimeError):
    """Raised instead of calling once too many calls have failed in a row."""


_client = None
_lock = threading.Lock()
_failures_in_a_row = 0


def client():
    """Create the Bedrock client once. boto3 clients are safe to share across threads."""
    global _client
    with _lock:
        if _client is None:
            # We do our own retries, so turn off boto's (otherwise the two stack up).
            config = Config(retries={"total_max_attempts": 1}, read_timeout=120)
            _client = boto3.Session(region_name=REGION).client("bedrock-runtime", config=config)
    return _client


def transient(e: Exception) -> bool:
    """Worth retrying? Throttling and server errors clear up; bad requests never will."""
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return code in TRANSIENT_CODES or status == 429 or status >= 500
    return isinstance(e, BotoCoreError)  # timeouts, dropped connections


def _record(ok: bool) -> None:
    global _failures_in_a_row
    with _lock:
        _failures_in_a_row = 0 if ok else _failures_in_a_row + 1


def _send(request: dict) -> tuple[dict, float]:
    """Call Bedrock with retries and the circuit breaker. Returns (response, seconds)."""
    if _failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
        raise CircuitOpen(f"{MAX_FAILURES_IN_A_ROW} LLM failures in a row: not calling")
    t0 = time.perf_counter()
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client().converse(**request)
            break
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1 or not transient(e):
                _record(ok=False)
                raise
            # 2, 4, 8 s plus a little jitter so parallel threads don't retry in lockstep.
            time.sleep(2 ** (attempt + 1) + random.random())
    _record(ok=True)
    return r, time.perf_counter() - t0


def converse(system: str, text: str, tool: dict | None = None, max_tokens: int = 1000) -> dict:
    """One LLM call. With `tool`, the model is forced to answer by filling that tool's
    JSON schema (structured output). Returns {"output", "tokens_in", "tokens_out", "seconds"}.
    """
    request = {"modelId": MODEL_ID, "system": [{"text": system}],
               "messages": [{"role": "user", "content": [{"text": text}]}],
               "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0}}
    if tool:
        request["toolConfig"] = {"tools": [{"toolSpec": tool}],
                                 "toolChoice": {"tool": {"name": tool["name"]}}}
    r, seconds = _send(request)
    content = r["output"]["message"]["content"]
    if tool:
        output = next((c["toolUse"]["input"] for c in content if "toolUse" in c), {})
    else:
        output = "".join(c.get("text", "") for c in content)
    return {"output": output, "tokens_in": r["usage"]["inputTokens"],
            "tokens_out": r["usage"]["outputTokens"], "seconds": seconds}


def converse_turn(system: str, messages: list, tools: list[dict], tool_choice: dict,
                  max_tokens: int = 1000) -> dict:
    """One turn of a multi-turn tool conversation (the agent loop in qa_agent.py).

    `tool_choice` is {"any": {}} (the model must call some tool, its choice) or
    {"tool": {"name": ...}} (this tool). Returns the assistant message and token counts.
    """
    request = {"modelId": MODEL_ID, "system": [{"text": system}], "messages": messages,
               "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
               "toolConfig": {"tools": [{"toolSpec": t} for t in tools],
                              "toolChoice": tool_choice}}
    r, seconds = _send(request)
    return {"message": r["output"]["message"], "tokens_in": r["usage"]["inputTokens"],
            "tokens_out": r["usage"]["outputTokens"], "seconds": seconds}


def cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
