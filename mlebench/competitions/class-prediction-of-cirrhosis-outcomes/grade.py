import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.competitions.utils import one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError

# The three status classes from your description
CLASSES = ["Status_C", "Status_CL", "Status_D"]


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """
    Validate submission format and prepare for log loss calculation
    """
    if set(submission.columns) != {"id"} | set(CLASSES):
        raise InvalidSubmissionError(
            "Submission DataFrame must have an 'id' column and columns: Status_C, Status_CL, Status_D"
        )
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            "Submission and answers DataFrames must have the same number of rows."
        )
    
    # According to the description, probabilities don't need to sum to 1 as they're rescaled
    # But they should still be valid probabilities (0-1 range)
    if not ((submission[CLASSES] >= 0) & (submission[CLASSES] <= 1)).all().all():
        raise InvalidSubmissionError(
            "All probabilities in submission DataFrame must be between 0 and 1."
        )

    assert set(answers.columns) == {"id"} | set(CLASSES), \
        "Answers DataFrame must have an 'id' column and columns for each status class."

    # apply_softmax=False since submissions should already be probabilities
    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission, answers, id_column="id", apply_softmax=False
    )

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)