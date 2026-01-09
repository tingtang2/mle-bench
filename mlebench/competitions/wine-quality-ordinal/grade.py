import pandas as pd
from sklearn.metrics import cohen_kappa_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using quadratic weighted kappa metric.
    """
    merged = submission.merge(answers[["Id", "quality"]], on="Id", suffixes=("_pred", "_true"))
    
    y_pred = merged["quality_pred"].values
    y_true = merged["quality_true"].values
    
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
