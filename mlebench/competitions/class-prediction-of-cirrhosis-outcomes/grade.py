import pandas as pd
from sklearn.metrics import log_loss

import numpy as np

from mlebench.competitions.utils import df_to_one_hot, one_hot_dfs_to_log_loss_inputs
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.utils import get_logger

logger = get_logger(__name__)


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'sample_id']:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{id_col}' column")

    # Determine class probability columns from submission.
    # Supported formats:
    # - Kaggle-style: id,C,CL,D
    # - Alternate: id,Status_C,Status_CL,Status_D
    # - Some agents output unnamed probability columns: 0,1,2,id (or id,0,1,2)
    sub_cols = list(submission.columns)
    if {"C", "CL", "D"}.issubset(set(sub_cols)):
        class_cols = ["C", "CL", "D"]
        submission_probs = submission[[id_col] + class_cols].copy()
    else:
        status_cols = [c for c in sub_cols if str(c).startswith("Status_")]
        if status_cols:
            mapped = {c: str(c).replace("Status_", "", 1) for c in status_cols}
            class_cols = [mapped[c] for c in status_cols]
            submission_probs = submission[[id_col] + status_cols].rename(columns=mapped).copy()
        else:
            # Fallback: any 3 non-id columns are treated as class probabilities.
            prob_cols = [c for c in sub_cols if c != id_col]
            if len(prob_cols) == 3:
                # If columns look like 0/1/2, sort numerically for stable mapping.
                if all(str(c).isdigit() for c in prob_cols):
                    prob_cols = sorted(prob_cols, key=lambda x: int(str(x)))
                mapped = {prob_cols[0]: "C", prob_cols[1]: "CL", prob_cols[2]: "D"}
                class_cols = ["C", "CL", "D"]
                submission_probs = submission[[id_col] + prob_cols].rename(columns=mapped).copy()
            else:
                raise InvalidSubmissionError(
                    "Submission must contain class probability columns. Expected either "
                    "['id','C','CL','D'] or ['id','Status_C','Status_CL','Status_D'] or "
                    "an id column plus exactly three probability columns (e.g., 0/1/2)."
                )

    # Build answers in one-hot format.
    if "Status" in answers.columns:
        answers_labels = answers[[id_col, "Status"]].copy()
        answers_onehot = df_to_one_hot(
            df=answers_labels, id_column=id_col, target_column="Status", classes=class_cols
        )
    else:
        # Fallback: already-one-hot answers
        status_target_cols = [c for c in answers.columns if c.startswith("Status_")]
        if status_target_cols:
            mapped = {c: c.replace("Status_", "", 1) for c in status_target_cols}
            answers_onehot = answers[[id_col] + status_target_cols].rename(columns=mapped).copy()
            # Align class columns order to submission
            missing = set(class_cols) - set(answers_onehot.columns)
            if missing:
                raise InvalidSubmissionError(
                    f"Answers is missing expected class columns after normalization: {sorted(missing)}"
                )
            answers_onehot = answers_onehot[[id_col] + class_cols]
        else:
            raise InvalidSubmissionError(
                "Could not find a target column in answers. Expected 'Status' or one-hot 'Status_*' columns."
            )

    # Ensure submission has required columns.
    for col in class_cols:
        if col not in submission_probs.columns:
            raise InvalidSubmissionError(f"Submission must have '{col}' column")

    # Ensure numeric predictions.
    for col in class_cols:
        submission_probs[col] = pd.to_numeric(submission_probs[col], errors="coerce")
    if submission_probs[class_cols].isna().any().any():
        raise InvalidSubmissionError("Submission contains NaNs/non-numeric values in class probability columns.")

    # Convert to one-hot format for log loss calculation (keep class column names).
    submission_onehot = submission_probs[[id_col] + class_cols].copy()
    answers_onehot = answers_onehot[[id_col] + class_cols].copy()

    # Robustness: some agents output an `id` column that does not match the expected ids
    # (e.g., normalized floats). If lengths match, fall back to aligning by row order by
    # overriding submission ids with the answer ids.
    try:
        sub_ids = set(submission_onehot[id_col].tolist())
        ans_ids = set(answers_onehot[id_col].tolist())
        if sub_ids != ans_ids:
            if len(submission_onehot) != len(answers_onehot):
                raise InvalidSubmissionError(
                    f"Submission and answers have different lengths ({len(submission_onehot)} vs {len(answers_onehot)}) "
                    "and ids do not match."
                )
            logger.warning(
                "Submission ids do not match answers ids; aligning by row order (overriding submission ids). "
                "This usually indicates the agent wrote an incorrect id column."
            )
            submission_onehot[id_col] = answers_onehot[id_col].to_numpy()
    except Exception as e:
        # If anything goes wrong in set conversion, treat as invalid submission.
        raise InvalidSubmissionError(f"Failed to validate/align ids: {e}")
    
    log_loss_inputs = one_hot_dfs_to_log_loss_inputs(
        submission_onehot, answers_onehot, id_column=id_col, apply_softmax=False
    )

    # If probabilities are not normalized, normalize (softmax-like) to avoid sklearn errors.
    y_pred = log_loss_inputs["y_pred"]
    row_sums = y_pred.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3) or (y_pred < 0).any():
        # Convert to a normalized distribution per row.
        y_pred = np.clip(y_pred, 1e-15, None)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        log_loss_inputs["y_pred"] = y_pred

    return log_loss_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_loss_inputs = prepare_for_metric(submission, answers)
    return log_loss(**log_loss_inputs)
