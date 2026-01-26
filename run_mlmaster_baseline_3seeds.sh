#!/bin/bash
# Run ML-Master baseline (no HPO enforcement) with 3 seeds, 9 competitions, 9 parallel workers, 2 hours per run

cd /home/amrutharao/mle/llms-for-mle-bench/mle-bench

python3 run_agent.py \
  --agent-id mlmaster \
  --competition-set experiments/splits/mlmaster_9comps_baseline.txt \
  --n-seeds 3 \
  --n-workers 9 \
  2>&1 | tee runs/mlmaster_baseline_3seeds_9comps_$(date +%Y%m%d_%H%M%S).log

echo "Run completed at $(date)"
