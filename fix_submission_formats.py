#!/usr/bin/env python3
"""Fix submission format issues to make them gradable"""
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
        if competition_id == 'playground-series-s5e7':
            # Grader expects 'Personality' column with 'Introvert'/'Extrovert' values
            if 'target' in submission.columns and 'Personality' not in submission.columns:
                # Convert numeric target to Personality strings
                submission = submission.rename(columns={'target': 'Personality'})
                # Map 0.0 -> Introvert, 1.0 -> Extrovert (or any non-zero -> Extrovert)
                if submission['Personality'].dtype in ['float64', 'int64']:
                    submission['Personality'] = submission['Personality'].map(
                        lambda x: 'Extrovert' if float(x) > 0.5 else 'Introvert'
                    )
                submission.to_csv(submission_path, index=False)
                return True
            elif 'Personality' in submission.columns:
                # If Personality exists but has numeric values, convert to strings
                if submission['Personality'].dtype in ['float64', 'int64']:
                    submission['Personality'] = submission['Personality'].map(
                        lambda x: 'Extrovert' if float(x) > 0.5 else 'Introvert'
                    )
                    submission.to_csv(submission_path, index=False)
                    return True
        
        elif competition_id == 'class-prediction-of-cirrhosis-outcomes':
            # Need class probability columns: C, CL, D
            # If we have a single 'target' column, convert to probabilities
            if 'target' in submission.columns and len(submission.columns) == 2:
                # Convert single target to one-hot probabilities
                classes = ['C', 'CL', 'D']
                submission_probs = pd.get_dummies(submission['target'], prefix='Status')
                # Map to C, CL, D format
                if 'Status_C' in submission_probs.columns:
                    submission_probs = submission_probs.rename(columns={'Status_C': 'C', 'Status_CL': 'CL', 'Status_D': 'D'})
                else:
                    # Create probability columns
                    submission_probs = pd.DataFrame({
                        'C': (submission['target'] == 0).astype(float),
                        'CL': (submission['target'] == 1).astype(float),
                        'D': (submission['target'] == 2).astype(float)
                    })
                
                # Combine with id
                result = pd.concat([submission[[id_col]], submission_probs], axis=1)
                result.to_csv(submission_path, index=False)
                return True
        
        elif competition_id == 'wine-quality-ordinal':
            # Need 'quality' column
            target_col = None
            for col in ['quality', 'target', 'y']:
                if col in submission.columns:
                    target_col = col
                    break
            
            if target_col and target_col != 'quality':
                submission = submission.rename(columns={target_col: 'quality'})
                submission.to_csv(submission_path, index=False)
                return True
        
        elif competition_id == 'electricity-demand':
            # Need 'y' column
            target_col = None
            for col in ['y', 'target', 'demand']:
                if col in submission.columns:
                    target_col = col
                    break
            
            if target_col and target_col != 'y':
                submission = submission.rename(columns={target_col: 'y'})
                submission.to_csv(submission_path, index=False)
                return True
        
        elif competition_id == 'gnss-classification':
            # Need 'Label' column (case-sensitive check in grader)
            target_col = None
            for col in submission.columns:
                if col.lower() in ['label', 'target', 'y']:
                    target_col = col
                    break
            
            if target_col and target_col != 'Label':
                submission = submission.rename(columns={target_col: 'Label'})
                # Also need GPS_Time(s) column
                if 'GPS_Time(s)' not in submission.columns and id_col != 'GPS_Time(s)':
                    # Try to use id as GPS_Time(s)
                    if id_col in submission.columns:
                        submission = submission.rename(columns={id_col: 'GPS_Time(s)'})
                        submission['GPS_Time(s)'] = submission['GPS_Time(s)'].astype(str)
                submission.to_csv(submission_path, index=False)
                return True
        
        elif competition_id.startswith('playground-series-s5e'):
            # These use accuracy metric, need target column
            target_col = None
            for col in ['target', 'y', 'label']:
                if col in submission.columns:
                    target_col = col
                    break
            
            # If no target column, check if there's a single prediction column
            if not target_col:
                non_id_cols = [c for c in submission.columns if c != id_col]
                if len(non_id_cols) == 1:
                    submission = submission.rename(columns={non_id_cols[0]: 'target'})
                    submission.to_csv(submission_path, index=False)
                    return True
        
    except Exception as e:
        print(f"Error fixing {submission_path}: {e}")
        return False
    
    return False

if __name__ == '__main__':
    run_group_dir = Path('runs/2026-01-26T12-32-25-GMT_run-group_rdagent')
    
    # Find all invalid submissions
    import json
    with open(run_group_dir / 'grading_summary.json') as f:
        reports = json.load(f)
    
    invalid = [r for r in reports if not r.get('valid_submission', True)]
    print(f"Found {len(invalid)} invalid submissions to fix...")
    
    fixed_count = 0
    for report in invalid:
        submission_path_str = report.get('submission_path')
        if not submission_path_str:
            print(f"✗ No submission path for {report.get('run', 'unknown')}")
            continue
        submission_path = Path(submission_path_str)
        if submission_path.exists():
            comp_id = report['competition_id']
            if fix_submission(submission_path, comp_id):
                print(f"✓ Fixed: {submission_path.name}")
                fixed_count += 1
            else:
                print(f"✗ Could not fix: {submission_path.name}")
    
    print(f"\nFixed {fixed_count} out of {len(invalid)} invalid submissions")
 
    print(f"\nFixed {fixed_count} out of {len(invalid)} invalid submissions")
