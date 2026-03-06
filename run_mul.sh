#!/bin/bash

#bash automation to run multiple seeds
# Run the command 10 times with 120 second gaps
for i in {1..10}; do
    echo "=========================================="
    echo "Starting run $i of 10 at $(date)"
    echo "=========================================="
    
    python run_agent.py --agent-id aide --competition-set experiments/splits/kaggle_short_list.txt --n-seeds 1 --n-workers 9
    
    # Don't sleep after the last iteration
    if [ $i -lt 10 ]; then
        echo "Run $i completed at $(date)"
        echo "Waiting 300  seconds (5  minutes) before next run..."
        sleep 120
    else
        echo "All 10 runs completed at $(date)"
    fi
done
