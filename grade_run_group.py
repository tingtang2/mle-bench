#!/usr/bin/env python3
"""Grade all runs in a run group and create grading_summary.json and scores_by_seed.txt"""
import json
import sys
from pathlib import Path
from datetime import datetime

from mlebench.grade import grade_csv
from mlebench.registry import registry as DEFAULT_REGISTRY
from mlebench.utils import get_timestamp

def main(run_group_dir: Path):
    run_group_dir = Path(run_group_dir)
    metadata_path = run_group_dir / "metadata.json"
    
    if not metadata_path.exists():
        print(f"Error: metadata.json not found at {metadata_path}")
        sys.exit(1)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    competition_reports = []
    registry = DEFAULT_REGISTRY
    
    print(f"Grading {len(metadata['runs'])} runs...")
    
    for run_id in metadata["runs"]:
        run_dir = run_group_dir / run_id
        comp_id = run_id.split("_")[0]
        submission_path = run_dir / "submission/submission.csv"
        
        if not submission_path.exists():
            print(f"Warning: No submission found for {run_id}, skipping...")
            # Create a report with null score
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
                "submission_exists": False,
                "valid_submission": False,
                "is_lower_better": None,
                "created_at": get_timestamp(),
                "submission_path": None,
                "run": run_id
            }
            competition_reports.append(report)
            continue
        
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
        "neurips-open-polymer-prediction-2025",  # Add any remaining competitions
    ]
    
    # Group reports by competition
    reports_by_comp = {}
    for report in competition_reports:
        comp_id = report["competition_id"]
        if comp_id not in reports_by_comp:
            reports_by_comp[comp_id] = []
        reports_by_comp[comp_id].append(report)
    
    # Sort each competition's reports by created_at to determine seed order
    for comp_id in reports_by_comp:
        reports_by_comp[comp_id].sort(key=lambda x: x.get("created_at", "") or "")
    
    # Determine number of seeds (max number of runs per competition)
    num_seeds = max(len(reports_by_comp.get(comp_id, [])) for comp_id in competition_order if comp_id in reports_by_comp)
    
    # Extract scores ordered by seed, then by competition
    scores_by_seed = []
    for seed_idx in range(num_seeds):
        for comp_id in competition_order:
            if comp_id in reports_by_comp and seed_idx < len(reports_by_comp[comp_id]):
                report = reports_by_comp[comp_id][seed_idx]
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
        print("Usage: python3 grade_run_group.py <run_group_dir>")
        sys.exit(1)
    
    main(Path(sys.argv[1]))
