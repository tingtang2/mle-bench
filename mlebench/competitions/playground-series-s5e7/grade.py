import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade a submission using accuracy score for binary classification.
    
    Args:
        submission: DataFrame with predictions (columns: id, Personality)
        answers: DataFrame with ground truth (columns: id, Personality)
    
    Returns:
        float: The accuracy score (higher is better)
    """
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, 
        answers=answers, 
        target_column="Personality", 
        id_column="id"
    )
    return accuracy_score(**accuracy_inputs)

