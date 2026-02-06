import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Determine id and target columns
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

    roc_auc_inputs = prepare_for_auroc_metric(
        submission=submission, answers=answers, id_col=id_col, target_col=target_col
    )
    return roc_auc_score(**roc_auc_inputs)
