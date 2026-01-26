#!/usr/bin/env python3
"""Grade all completed runs in a run group directory directly (without metadata.json)"""
import json
import sys
from pathlib import Path
from collections import defaultdict

from mlebench.grade import grade_csv
from mlebench.registry import registry as DEFAULT_REGISTRY
from mlebench.utils import get_timestamp

def main(run_group_dir: Path):
    run_group_dir = Path(run_group_dir)
    
    if not run_group_dir.exists():
        print(f"Error: Directory not found at {run_group_dir}")
        sys.exit(1)
    
    # Find all submission.csv files
    submission_paths = list(run_group_dir.glob("*/submission/submission.csv"))
    
    if not submission_paths:
        print(f"No submission files found in {run_group_dir}")
        sys.exit(1)
    
    print(f"Found {len(submission_paths)} submission files...")
    
    competition_reports = []
    registry = DEFAULT_REGISTRY
    
    # Grade each submission
    for submission_path in sorted(submission_paths):
        run_dir = submission_path.parent.parent
        run_id = run_dir.name
        comp_id = run_id.split("_")[0]
        
        try:
            competition = registry.get_competition(comp_id)
            report = grade_csv(submission_path, competition)
            report_dict = report.to_dict()
            report_dict["submission_path"] = str(submission_path)
            report_dict["run"] = run_id
            report_dict["created_at"] = get_timestamp()
            competition_reports.append(report_dict)
        except Exception as e:
            print(f"Warning: Error grading {run_id}: {e}")
            # Create a report with null score on error
            try:
                competition = registry.get_competition(comp_id)
                report = {
                    "competition_id": comp_id,
                    "score": None,
                    "gold_threshold": None,
                    "silver_threshold": None,
                    "bronze_threshold": None,
                    "median_threshold": None,
                    "any_medal": False,
                    "gold_medal": False,
                    "silver_medal": False,
                    "bronze_medal": False,
                    "above_median": False,
                    "submission_exists": True,
                    "valid_submission": False,
                    "is_lower_better": None,
                    "created_at": get_timestamp(),
                    "submission_path": str(submission_path),
                    "run": run_id
                }
                competition_reports.append(report)
            except Exception as e2:
                print(f"Error: Failed to create report for {run_id}: {e2}")
                continue
    
    # Save grading_summary.json
    grading_summary_path = run_group_dir / "grading_summary.json"
    with open(grading_summary_path, "w") as f:
        json.dump(competition_reports, f, indent=2)
    print(f"Saved grading_summary.json with {len(competition_reports)} reports")
    
    # Create scores_by_seed.txt
    # Define competition order (as specified by user)
    competition_order = [
        "class-prediction-of-cirrhosis-outcomes",
        "electricity-demand",
        "gnss-classification",
        "playground-series-s5e3",
        "playground-series-s5e6",
        "playground-series-s5e7",
        "playground-series-s5e8",
        "spaceship-titanic",
        "wine-quality-ordinal",
        "neurips-open-polymer-prediction-2025",
    ]
    
    # Group reports by competition and sort by created_at within each competition
    reports_by_comp = defaultdict(list)
    for report in competition_reports:
        comp_id = report["competition_id"]
        reports_by_comp[comp_id].append(report)
    
    # Sort each competition's reports by created_at to determine seed order
    for comp_id in reports_by_comp:
        reports_by_comp[comp_id].sort(key=lambda x: x.get("created_at", "") or "")
    
    # Extract scores ordered by competition first, then by seed
    # This means: Competition 1 (seed 1, seed 2, seed 3), Competition 2 (seed 1, seed 2, seed 3), etc.
    scores_by_seed = []
    for comp_id in competition_order:
        if comp_id in reports_by_comp:
            comp_reports = reports_by_comp[comp_id]
            # Sort by created_at to maintain seed order (seed 1, then seed 2, then seed 3)
            comp_reports.sort(key=lambda x: x.get("created_at", "") or "")
            for report in comp_reports:
                score = report.get("score")
                scores_by_seed.append(score if score is not None else "null")
    
    # Write scores_by_seed.txt
    scores_file_path = run_group_dir / "scores_by_seed.txt"
    with open(scores_file_path, "w") as f:
        for score in scores_by_seed:
            f.write(f"{score}\n")
    print(f"Saved scores_by_seed.txt with {len(scores_by_seed)} scores")
    
    print("Grading complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 grade_run_group_direct.py <run_group_dir>")
        sys.exit(1)
    
    main(Path(sys.argv[1]))
