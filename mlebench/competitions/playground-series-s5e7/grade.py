import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Find id and target columns
    id_col = None
    target_col = None
    
    for col in ['id', 'Id', 'ID', 'sample_id']:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]
    
    for col in ['target', 'y', 'label']:
        if col in answers.columns:
            target_col = col
            break
    if target_col is None:
        target_col = [c for c in answers.columns if c != id_col][-1]
    
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column=target_col, id_column=id_col
    )
    return accuracy_score(**accuracy_inputs)
