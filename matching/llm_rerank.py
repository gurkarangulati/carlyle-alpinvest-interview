"""Optional LLM re-rank for close calls: Claude Haiku 4.5 on Amazon Bedrock.

Used by evaluate.py --llm-rerank. Only low-margin queries come here: the hybrid's
top two companies are within LLM_MARGIN points, so string similarity cannot tell
them apart, but world knowledge sometimes can ("Alphabet" vs "Alphabet Holdings").

Credentials: boto3 reads the AWS profile from the AWS_PROFILE environment variable.
"""

import threading
import time

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"
PRICE_IN, PRICE_OUT = 1.00, 5.00  # USD per 1M tokens, list price. Verify before quoting.
LLM_MARGIN = 5.0  # send a query when its top two scores are this close
LLM_MIN_SCORE = 70.0  # ...and the best one is at least plausible
MAX_FAILURES_IN_A_ROW = 10  # circuit breaker: a hard error will not fix itself

SYSTEM = """You resolve company names to SEC registrants.
You get a query name and a short list of candidate companies, each with a CIK,
its current name, and the alias that matched (which may be a former name or ticker).
Pick the candidate that is the same company as the query. A former name of the
company counts as the same company. A parent, subsidiary, or a different company
with a similar name does not. If no candidate is clearly the same company, pick "none"."""


def needs_llm(cands: list[dict]) -> bool:
    """Low margin: two plausible companies, too close to call on string similarity."""
    if len(cands) < 2 or cands[0]["score"] < LLM_MIN_SCORE:
        return False
    return cands[0]["score"] - cands[1]["score"] < LLM_MARGIN


def transient(e: Exception) -> bool:
    """Worth retrying? Throttling (429) and server errors (5xx) clear up; other 4xx never will."""
    if isinstance(e, ClientError):
        code = e.response["Error"].get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return status == 429 or status >= 500 or "Throttl" in code
    return False  # BotoCoreError (timeout, no credentials): count it as a failure


class LLMReranker:
    """Asks the model to pick one candidate CIK or "none". Tracks spend; stops at a budget."""

    def __init__(self, budget_usd: float = 1.50):
        # No retries inside boto3: we retry ourselves, and only on errors worth retrying.
        cfg = Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=30)
        self.client = boto3.Session(region_name=REGION).client("bedrock-runtime", config=cfg)
        self.budget_usd = budget_usd
        self.cost_usd = 0.0
        self.calls = self.invalid = self.failures_in_a_row = 0
        self.lock = threading.Lock()  # evaluate.py calls from several threads

    def _ask(self, query: str, cands: list[dict]) -> tuple[str, int, int]:
        allowed = [str(c["cik"]) for c in cands] + ["none"]
        listing = "\n".join(f"- CIK {c['cik']}: {c['name']} (matched {c['alias_kind']}: "
                            f"{c['alias']})" for c in cands)
        # Forcing a tool whose input is an enum = structured output on Bedrock.
        tool = {"toolSpec": {
            "name": "pick_company", "description": "Record the matching CIK, or none.",
            "inputSchema": {"json": {"type": "object", "required": ["cik"], "properties": {
                "cik": {"type": "string", "enum": allowed}}}}}}
        r = self.client.converse(
            modelId=MODEL_ID, system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [
                {"text": f"Query: {query}\nCandidates:\n{listing}"}]}],
            inferenceConfig={"maxTokens": 50, "temperature": 0},
            toolConfig={"tools": [tool], "toolChoice": {"tool": {"name": "pick_company"}}})
        block = next((c for c in r["output"]["message"]["content"] if "toolUse" in c), None)
        answer = str(block["toolUse"]["input"].get("cik", "")) if block else ""  # "" = invalid
        return answer, r["usage"]["inputTokens"], r["usage"]["outputTokens"]

    def pick(self, query: str, cands: list[dict]) -> tuple[int | None, str]:
        """Returns (cik or None, status). Status: ok, none, invalid, error, breaker, budget."""
        with self.lock:
            if self.failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                return None, "breaker"
            if self.cost_usd >= self.budget_usd:
                return None, "budget"
        for attempt in range(4):
            try:
                answer, tin, tout = self._ask(query, cands)
                break
            except (ClientError, BotoCoreError) as e:
                if attempt == 3 or not transient(e):
                    with self.lock:
                        self.failures_in_a_row += 1
                    print(f"  LLM call failed: {type(e).__name__}: {str(e)[:120]}")
                    return None, "error"
                time.sleep(2 * 2 ** attempt)  # 2, 4, 8 s: throttling clears in seconds
        with self.lock:
            self.failures_in_a_row = 0
            self.calls += 1
            self.cost_usd += (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
        # Bedrock does not strictly enforce the enum, so check the answer in code.
        if answer == "none":
            return None, "none"
        if answer not in {str(c["cik"]) for c in cands}:
            with self.lock:
                self.invalid += 1
            return None, "invalid"
        return int(answer), "ok"
