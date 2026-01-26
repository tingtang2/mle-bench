import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.metrics import mean_average_precision_at_k


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'user_id', 'customer_id']:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]
    
    # Find prediction column
    pred_col = None
    for col in ['prediction', 'target', 'items', 'recommendations']:
        if col in answers.columns:
            pred_col = col
            break
    if pred_col is None:
        pred_col = [c for c in answers.columns if c != id_col][-1]
    
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{id_col}' column")
    if pred_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{pred_col}' column")
    
    # Sort by id
    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)
    
    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError("Submission and answers must have matching ids")
    
    # Convert predictions to lists (space-separated or comma-separated)
    y_true = answers[pred_col].astype(str).str.split().apply(set).tolist()
    y_pred = submission[pred_col].astype(str).str.split().apply(list).tolist()
    
    return {"actual": y_true, "predicted": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    prepped = prepare_for_metric(submission, answers)
    return mean_average_precision_at_k(
        actual=prepped["actual"], predicted=prepped["predicted"], k=3
    )
