# The analyst agent for Amazon Bedrock AgentCore Runtime.
# AgentCore runs ARM64 (Graviton) containers that serve POST /invocations and GET /ping
# on port 8080. Build context = sprint/ml (the parent folder), so the image can take the
# shared modules it reuses:
#   docker build --platform linux/arm64 -f agent/Dockerfile -t analyst-agent .
# Only the files below go in (see Dockerfile.dockerignore). No data is baked in: in the
# cloud the agent reads the lake through Athena (LAKE_BACKEND=athena).
FROM --platform=linux/arm64 python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCKER_CONTAINER=1 \
    LAKE_BACKEND=athena \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY agent/requirements.txt agent/requirements.txt
RUN pip install -r agent/requirements.txt

# Reused modules: the name matcher (+ its tuned thresholds) and, from extraction/,
# the LLM client, the safe calculator and the verify helpers, plus common.py they import.
COPY matching/__init__.py matching/matcher.py matching/
COPY matching/results/hybrid_thresholds.json matching/results/
COPY extraction/llm.py extraction/qa_tools.py extraction/qa_agent.py extraction/common.py extraction/
COPY agent/*.py agent/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 agent && chown -R agent /app
USER agent

EXPOSE 8080
CMD ["python", "agent/app.py"]
