#!/bin/bash
# Run ML-Master with 3 seeds, 9 competitions, 9 parallel workers, 2 hours per run

cd "$(dirname "$0")"

# Load .env if present (e.g. OPENAI_API_KEY)
[[ -f .env ]] && set -a && source .env && set +a

# Use project venv if present (has docker + mle-bench deps)
PYTHON="${PWD}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3

$PYTHON run_agent.py \
  --agent-id mlmaster \
  --competition-set experiments/splits/mlmaster_9comps_baseline.txt \
  --n-seeds 3 \
  --n-workers 9 \
  2>&1 | tee runs/mlmaster_9comps_3seeds_$(date +%Y%m%d_%H%M%S).log

echo "Run completed at $(date)"
