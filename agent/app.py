"""The AgentCore Runtime entrypoint: the agent as an HTTP service on port 8080.

Local:   uv run python agent/app.py
         curl -s localhost:8080/invocations -H 'Content-Type: application/json' \\
              -d '{"prompt": "Which fund has the most technology exposure?"}'
         curl -s localhost:8080/ping
Cloud:   the same file is the container's command; AgentCore Runtime calls
         POST /invocations for every invoke_agent_runtime and GET /ping for health.

BedrockAgentCoreApp (bedrock-agentcore SDK, checked against version 1.24) is a small
Starlette app that provides exactly those two routes. `@app.entrypoint` registers our
function; the SDK runs a plain (non-async) function in a thread pool, so the blocking
Bedrock and Athena calls do not stall /ping. Whatever we return is sent back as JSON.

The lake engine comes from LAKE_BACKEND: "local" on a laptop, "athena" in the cloud
(set by the CDK AgentStack together with ATHENA_DATABASE / ATHENA_WORKGROUP / ATHENA_OUTPUT).
"""

import logging
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402

from agent.graph import ask  # noqa: E402

MAX_PROMPT_CHARS = 2000
log = logging.getLogger("analyst-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """{"prompt": "..."} -> {"answer", "answer_value", "citations", "status", "trace", ...}."""
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return {"status": "error", "reason": 'send {"prompt": "<question>"}'}
    if len(prompt) > MAX_PROMPT_CHARS:
        return {"status": "error", "reason": f"prompt longer than {MAX_PROMPT_CHARS} characters"}
    result = ask(prompt.strip())
    # One structured log line per question: what CloudWatch keeps (no conversation text).
    log.info("question status=%s tool_calls=%s tokens_in=%s tokens_out=%s cost_usd=%s "
             "bytes_scanned=%s seconds=%s queries=%s", result["status"], result["tool_calls"],
             result["tokens_in"], result["tokens_out"], result["cost_usd"],
             result["bytes_scanned"], result["seconds"], list(result["queries"]))
    return result


if __name__ == "__main__":
    app.run()  # 127.0.0.1:8080 on a laptop; 0.0.0.0 inside a container (DOCKER_CONTAINER=1)
