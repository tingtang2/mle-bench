#!/bin/bash
# Test run for ML-Master with hyperparameter tuning enabled, 12 minutes
cd /home/amrutharao/mle/llms-for-mle-bench/mle-bench

# Override TIME_LIMIT_SECS for test (720 seconds = 12 minutes)
export TIME_LIMIT_SECS=720

python3 run_agent.py \
  --agent-id mlmaster \
  --competition-set experiments/splits/spaceship-titanic.txt \
  --n-seeds 1 \
  --n-workers 1
