#!/bin/bash
#
# Run RD-Agent on MLE-bench competitions.
#
# Usage:
#   bash run_rdagent.sh                          # default: spaceship-titanic, rdagent preset
#   bash run_rdagent.sh spaceship-titanic         # specific competition
#   bash run_rdagent.sh spaceship-titanic light   # use rdagent/light preset (15 min)
#   bash run_rdagent.sh ALL                       # run all prepared competitions
#
# Prerequisites:
#   - Docker image built: bash build_agent.sh agents/rdagent/Dockerfile
#   - ~/.aide_env has OPENAI_API_KEY
#   - ~/kaggle.json has Kaggle credentials

set -euo pipefail
cd "$(dirname "$0")"

# ── Args ──────────────────────────────────────────────────────────────────
COMPETITION="${1:-spaceship-titanic}"
PRESET="${2:-rdagent}"
AGENT_ID="rdagent/${PRESET}"
[[ "$PRESET" == "rdagent" ]] && AGENT_ID="rdagent"

# ── Secrets ───────────────────────────────────────────────────────────────
export OPENAI_API_KEY=$(grep OPENAI_API_KEY ~/.aide_env | cut -d= -f2)
export KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('$HOME/kaggle.json'))['username'])")
export KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('$HOME/kaggle.json'))['key'])")

# ── Competition list ──────────────────────────────────────────────────────
if [[ "$COMPETITION" == "ALL" ]]; then
  COMP_FILE="experiments/splits/prepared.txt"
else
  COMP_FILE=$(mktemp /tmp/rdagent_comp_XXXX.txt)
  echo "$COMPETITION" > "$COMP_FILE"
fi

echo "=============================="
echo "Agent:       $AGENT_ID"
echo "Competition: $COMPETITION"
echo "Comp file:   $COMP_FILE"
echo "=============================="

# ── Run ───────────────────────────────────────────────────────────────────
python run_agent.py \
  --agent-id "$AGENT_ID" \
  --competition-set "$COMP_FILE" \
  --n-seeds 1 \
  --container-config agents/rdagent/config/container_config.json \
  --run-dir ./runs

# ── Grade ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Grading ==="
LATEST_RUN=$(ls -dt runs/*_run-group_rdagent/ 2>/dev/null | head -1)
if [[ -z "$LATEST_RUN" ]]; then
  echo "No run directory found."
  exit 1
fi

METADATA="${LATEST_RUN}/metadata.json"
SUBMISSION_JSONL="${LATEST_RUN}/submission.jsonl"
GRADING_OUTPUT="${LATEST_RUN}/grading_output"

python experiments/make_submission.py \
  --metadata "$METADATA" \
  --output "$SUBMISSION_JSONL" \
  --rel-log-path "logs/entrypoint.log" \
  --rel-code-path "code/"

mlebench grade \
  --submission "$SUBMISSION_JSONL" \
  --output-dir "$GRADING_OUTPUT"

echo ""
echo "Grading results saved to: $GRADING_OUTPUT"
ls -t "$GRADING_OUTPUT"/*_grading_report.json 2>/dev/null | head -1 | xargs cat 2>/dev/null || true
