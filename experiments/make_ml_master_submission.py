import json
from pathlib import Path

workspace_dir = Path("/home/aa3320/llms-for-mle-bench/ML-Master/workspaces/run")
batch_results_dir = Path("/home/aa3320/llms-for-mle-bench/ML-Master/batch_results/kaggle_short_list_10seeds_20260303_194507")
output = Path("/home/aa3320/llms-for-mle-bench/mle-bench/runs/submission.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as f:
    for run_dir in sorted(workspace_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Extract comp_id and seed from e.g. "wine-quality-ordinal_seed10_mcts_comp_validcheck_[cpu-0-21]"
        parts = run_dir.name.split("_seed")
        if len(parts) < 2:
            continue
        comp_id = parts[0]
        seed = parts[1].split("_")[0]

        submission = run_dir / "best_submission" / "submission.csv"
        solution = run_dir / "best_solution" / "solution.py"
        log = batch_results_dir / f"{comp_id}_seed{seed}_run.log"

        f.write(json.dumps({
            "competition_id": comp_id,
            "submission_path": str(submission) if submission.exists() else None,
            "code_path": str(solution) if solution.exists() else None,
            "logs_path": str(log) if log.exists() else None,
        }) + "\n")

print(f"Done: {output}")
