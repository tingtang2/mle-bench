#!/bin/bash
# Run AIDE with 5 seeds, 9 competitions, 9 parallel workers, 2 hours each

cd "$(dirname "$0")"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set!"
    echo "Please export it first: export OPENAI_API_KEY='your-key'"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="runs/aide_9comps_5seeds_2hr_${TIMESTAMP}.log"

echo "Starting aide run at $(date)"
echo "Log file: $LOG_FILE"

python3 run_agent.py \
  --agent-id aide \
  --competition-set experiments/splits/mlmaster_9comps_baseline.txt \
  --n-seeds 5 \
  --n-workers 9 \
  2>&1 | tee "${LOG_FILE}"

echo "Run completed at $(date)" | tee -a "${LOG_FILE}"
