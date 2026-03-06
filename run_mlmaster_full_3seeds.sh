#!/bin/bash
# Run ML-Master with 3 seeds, 10 competitions, 10 parallel workers, 2 hours per run

cd "$(dirname "$0")"

python3 run_agent.py \
  --agent-id mlmaster \
  --competition-set experiments/splits/mlmaster_10comps.txt \
  --n-seeds 3 \
  --n-workers 10 \
  2>&1 | tee runs/mlmaster_3seeds_10comps_$(date +%Y%m%d_%H%M%S).log

echo "Run completed at $(date)"
