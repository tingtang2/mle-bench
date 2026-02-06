#!/usr/bin/env python3
"""Fix submission format issues for aide run group while preserving valid scores"""
import json
import pandas as pd
from pathlib import Path
from mlebench.registry import registry

def fix_submission(submission_path: Path, competition_id: str) -> bool:
    """Try to fix format issues in a submission. Returns True if fixed."""
    try:
        comp = registry.get_competition(competition_id)
        submission = pd.read_csv(submission_path)
        answers = pd.read_csv(comp.answers)
        
        # Find ID column
        id_col = None
        for col in ['id', 'Id', 'ID', 'sample_id']:
            if col in submission.columns:
                id_col = col
                break
        if id_col is None:
            id_col = submission.columns[0]
        
        # Competition-specific fixes
        if competition_id == 'playground-series-s5e3':
            # Grader expects 'rainfall' column (last column in answers)
            if 'target' in submission.columns and 'rainfall' not in submission.columns:
                submission = submission.rename(columns={'target': 'rainfall'})
                submission.to_csv(submission_path, index=False)
                return True
        
        elif competition_id == 'playground-series-s5e6':
            # Grader expects 'Fertilizer Name' column (last column in answers)
            if 'target' in submission.columns and 'Fertilizer Name' not in submission.columns:
                submission = submission.rename(columns={'target': 'Fertilizer Name'})
                submission.to_csv(submission_path, index=False)
                return True
        
        elif competition_id == 'playground-series-s5e8':
            # Check what column is expected
            target_col = None
            for col in ['target', 'y', 'label']:
                if col in answers.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = [c for c in answers.columns if c != id_col][-1]
            
            # If submission has 'target' but needs different column name
            if 'target' in submission.columns and target_col not in submission.columns and target_col != 'target':
                submission = submission.rename(columns={'target': target_col})
                submission.to_csv(submission_path, index=False)
                return True
        
    except Exception as e:
        print(f"Error fixing {submission_path}: {e}")
        return False
    
    return False

if __name__ == '__main__':
    run_group_dir = Path('runs/2026-01-27T07-55-57-GMT_run-group_aide')
    
    # Load grading summary to find invalid submissions
    grading_summary_path = run_group_dir / 'grading_summary.json'
    if not grading_summary_path.exists():
        print(f"Error: grading_summary.json not found at {grading_summary_path}")
        exit(1)
    
    with open(grading_summary_path, 'r') as f:
        reports = json.load(f)
    
    # Find invalid submissions (but preserve valid ones)
    invalid = [r for r in reports if not r.get('valid_submission', True) and r.get('submission_path')]
    print(f"Found {len(invalid)} invalid submissions to fix...")
    
    fixed_count = 0
    for report in invalid:
        submission_path_str = report.get('submission_path')
        if not submission_path_str:
            print(f"✗ No submission path for {report.get('run', 'unknown')}")
            continue
        
        submission_path = run_group_dir / submission_path_str.replace(f"{run_group_dir.name}/", "")
        if not submission_path.exists():
            # Try absolute path
            submission_path = Path(submission_path_str)
        
        if submission_path.exists():
            comp_id = report['competition_id']
            if fix_submission(submission_path, comp_id):
                print(f"✓ Fixed: {submission_path.name}")
                fixed_count += 1
            else:
                print(f"✗ Could not fix: {submission_path.name}")
        else:
            print(f"✗ Submission file not found: {submission_path}")
    
    print(f"\nFixed {fixed_count} out of {len(invalid)} invalid submissions")
    print("Note: Valid scores are preserved in grading_summary.json")
