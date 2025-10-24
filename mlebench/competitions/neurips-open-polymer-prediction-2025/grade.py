import numpy as np
import pandas as pd

from mlebench.grade_helpers import InvalidSubmissionError

# The five polymer properties
PROPERTIES = ["density", "Tc", "Tg", "Rg", "FFV"]


def weighted_mae(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Calculate weighted Mean Absolute Error (wMAE) as defined in the competition.
    
    wMAE = 1/X * sum(sum(w_i * |y_i - y_pred_i|)) for X in X and i in I(X)
    
    where the reweighting factor is:
    w_i = (1/r_i) * (K*sqrt(1/n_i))/sum(sqrt(1/n_j)) for j in 1 to K
    
    Args:
        submission: DataFrame with 'id' and property predictions
        answers: DataFrame with 'id' and ground truth property values
    
    Returns:
        float: The weighted MAE score
    """
    # Validate columns
    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'id' column")
    for prop in PROPERTIES:
        if prop not in submission.columns:
            raise InvalidSubmissionError(f"Submission must contain the '{prop}' column")
    
    if "id" not in answers.columns:
        raise InvalidSubmissionError("Answers must contain the 'id' column")
    for prop in PROPERTIES:
        if prop not in answers.columns:
            raise InvalidSubmissionError(f"Answers must contain the '{prop}' column")
    
    # Sort and align by id
    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)
    
    # Check that IDs match
    if not submission["id"].equals(answers["id"]):
        raise InvalidSubmissionError("Submission and answers must have matching IDs")
    
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows but answers has {len(answers)} rows"
        )
    
    K = len(PROPERTIES)  # Total number of properties (5)
    
    # Calculate weights for each property
    weights = {}
    sqrt_inv_n_sum = 0
    
    for prop in PROPERTIES:
        # Count non-NaN values for this property
        y_true_prop = answers[prop].values
        valid_mask = ~np.isnan(y_true_prop)
        n_i = valid_mask.sum()
        
        if n_i == 0:
            # If no valid values for this property, skip it
            weights[prop] = 0
            continue
        
        # Calculate range r_i
        y_true_valid = y_true_prop[valid_mask]
        r_i = y_true_valid.max() - y_true_valid.min()
        
        # Avoid division by zero in range
        if r_i == 0:
            r_i = 1.0  # Use 1.0 if all values are the same
        
        # Store intermediate values for weight calculation
        weights[prop] = {
            'n_i': n_i,
            'r_i': r_i,
            'sqrt_inv_n': np.sqrt(1.0 / n_i)
        }
        sqrt_inv_n_sum += weights[prop]['sqrt_inv_n']
    
    # Calculate final weights: w_i = (1/r_i) * (K*sqrt(1/n_i))/sum(sqrt(1/n_j))
    final_weights = {}
    for prop in PROPERTIES:
        if isinstance(weights[prop], dict):
            scale_norm = 1.0 / weights[prop]['r_i']
            weight_norm = (K * weights[prop]['sqrt_inv_n']) / sqrt_inv_n_sum
            final_weights[prop] = scale_norm * weight_norm
        else:
            final_weights[prop] = 0
    
    # Calculate weighted MAE
    total_weighted_error = 0
    total_samples = len(submission)
    
    for prop in PROPERTIES:
        if final_weights[prop] == 0:
            continue
        
        y_true = answers[prop].values
        y_pred = submission[prop].values
        
        # Only consider non-NaN values in ground truth
        valid_mask = ~np.isnan(y_true)
        
        if valid_mask.sum() == 0:
            continue
        
        # Calculate absolute errors for valid samples
        abs_errors = np.abs(y_true[valid_mask] - y_pred[valid_mask])
        
        # Weighted sum of errors for this property
        weighted_errors = final_weights[prop] * abs_errors
        total_weighted_error += weighted_errors.sum()
    
    # Average over all samples
    wmae = total_weighted_error / total_samples
    
    return wmae


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade a submission using weighted MAE.
    
    Args:
        submission: DataFrame with predictions
        answers: DataFrame with ground truth
    
    Returns:
        float: The weighted MAE score (lower is better)
    """
    return weighted_mae(submission, answers)

