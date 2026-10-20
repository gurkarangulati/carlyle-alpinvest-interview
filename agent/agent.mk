# The analyst agent: make targets. Included by sprint/ml/Makefile (`include agent/agent.mk`);
# uses its PY, AWS, PROFILE, REGION and check-aws. Run from sprint/ml.
#
#   local:    make ask Q="..."   make agent-serve   make eval-agent
#   deploy:   make deploy-lake   make deploy-agent  (or make deploy-all)
#   validate: make ask-aws Q="..."   make eval-agent TARGET=aws
#   teardown: make teardown-all  (agent first, then lake; see agent/deploy/TEARDOWN.md)

TARGET    ?= local
AGENT_CDK := cd agent/deploy && $(AWS) JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1 npx aws-cdk@latest

.PHONY: ask agent-serve eval-agent agent-test agent-synth deploy-lake deploy-agent deploy-all \
	    ask-aws teardown-agent teardown-lake teardown-all check-q check-lake

check-q:
	@test -n "$(Q)" || { echo 'Set Q="your question"'; exit 1; }
check-lake:
	@test -f data/lake/silver/companies/data.parquet || { echo "Build the lake first (data/lake/silver missing)"; exit 1; }

# ---- local ----
ask: check-aws check-q  ## ask the agent one question against the local lake (Bedrock for the LLM)
	$(AWS) $(PY) agent/graph.py --question "$(Q)"

agent-serve: check-aws  ## run the AgentCore app locally: curl localhost:8080/invocations
	$(AWS) $(PY) agent/app.py

eval-agent: check-aws  ## golden questions: TARGET=local (default) or TARGET=aws
	mkdir -p agent/results
	$(AWS) $(PY) agent/evaluate.py --target $(TARGET) 2>&1 | tee agent/results/eval_$(TARGET)_$(STAMP).log

agent-test:  ## agent unit tests (SQL guard, backend, verify, call cap, scoring)
	$(PY) -m pytest -q tests/test_agent.py

agent-synth:  ## CloudFormation for both stacks, no AWS calls (agent/deploy/cdk.out)
	$(AGENT_CDK) synth --quiet

# ---- deploy (read agent/deploy/TEARDOWN.md first) ----
deploy-lake: check-aws check-lake  ## S3 bucket + Parquet upload + Glue tables + Athena workgroup
	$(AGENT_CDK) deploy AnalystLake --require-approval broadening

deploy-agent: check-aws  ## build the ARM64 image, push it, create the AgentCore Runtime (needs the lake)
	$(AGENT_CDK) deploy AnalystAgent --exclusively --require-approval broadening

deploy-all: check-aws check-lake  ## both stacks, lake first
	$(AGENT_CDK) deploy --all --require-approval broadening

# ---- validate ----
ask-aws: check-aws check-q  ## ask the DEPLOYED agent (ARN from the stack outputs)
	$(AWS) $(PY) agent/invoke.py --question "$(Q)"

# ---- teardown: cdk destroy, then what it leaves (TEARDOWN.md) ----
teardown-agent: check-aws  ## destroy AnalystAgent, then its log groups and its image in the CDK ECR repo
	@RID=$$($(AWS) aws cloudformation describe-stacks --stack-name AnalystAgent \
	    --query "Stacks[0].Outputs[?OutputKey=='RuntimeId'].OutputValue" --output text 2>/dev/null); \
	URI=""; \
	if [ -n "$$RID" ] && [ "$$RID" != "None" ]; then \
	  URI=$$($(AWS) aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$$RID" \
	    --query agentRuntimeArtifact.containerConfiguration.containerUri --output text 2>/dev/null); \
	fi; \
	($(AGENT_CDK) destroy AnalystAgent --force) || exit 1; \
	for g in $$($(AWS) aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/analyst_agent \
	    --query 'logGroups[].logGroupName' --output text); do \
	  [ "$$g" = "None" ] && continue; $(AWS) aws logs delete-log-group --log-group-name "$$g" && echo "deleted log group $$g"; \
	done; \
	if [ -n "$$URI" ] && [ "$$URI" != "None" ]; then \
	  REPO=$${URI#*/}; REPO=$${REPO%%:*}; TAG=$${URI##*:}; \
	  $(AWS) aws ecr batch-delete-image --repository-name "$$REPO" --image-ids imageTag="$$TAG" \
	    --query 'imageIds[].imageTag' --output text && echo "deleted image $$TAG from $$REPO"; \
	fi

teardown-lake: check-aws  ## destroy AnalystLake (bucket emptied automatically), then its Lambda log groups
	$(AGENT_CDK) destroy AnalystLake --force
	@for g in $$($(AWS) aws logs describe-log-groups --log-group-name-prefix /aws/lambda/AnalystLake \
	    --query 'logGroups[].logGroupName' --output text); do \
	  [ "$$g" = "None" ] && continue; $(AWS) aws logs delete-log-group --log-group-name "$$g" && echo "deleted log group $$g"; \
	done

teardown-all: teardown-agent teardown-lake  ## agent first (it imports the lake's names), then lake
