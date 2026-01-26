#!/bin/bash
# Run AIDE with 3 seeds, 9 competitions, 9 parallel workers, 2 hours each

cd /home/amrutharao/mle/llms-for-mle-bench/mle-bench

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="runs/aide_9comps_3seeds_2hr_${TIMESTAMP}.log"

python3 run_agent.py \
  --agent-id aide \
  --competition-set experiments/splits/mlmaster_9comps_baseline.txt \
  --n-seeds 3 \
  --n-workers 9 \
  2>&1 | tee "${LOG_FILE}"

echo "Run completed at $(date)" | tee -a "${LOG_FILE}"
