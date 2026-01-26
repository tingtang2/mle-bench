import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric
from mlebench.grade_helpers import InvalidSubmissionError


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
    
    # Target column: match common names case-insensitively (e.g., "Label")
    target_priority = ["target", "y", "label", "class"]
    answers_cols_by_lower = {c.lower(): c for c in answers.columns}
    for key in target_priority:
        if key in answers_cols_by_lower:
            target_col = answers_cols_by_lower[key]
            break
    if target_col is None:
        target_col = [c for c in answers.columns if c != id_col][-1]

    # This dataset can contain multiple rows per GPS timestamp (e.g., one per satellite)
    # and may also include empty rows. For grading, reduce answers to a single label per id,
    # then align submission/answers on the intersection of ids.
    submission_simple = submission[[id_col, target_col]].dropna(subset=[id_col]).copy()
    answers_simple = answers[[id_col, target_col]].dropna(subset=[id_col]).copy()

    if len(submission_simple) == 0:
        raise InvalidSubmissionError("Submission has no valid ids after dropping missing values.")
    if len(answers_simple) == 0:
        raise InvalidSubmissionError("Answers has no valid ids after dropping missing values.")

    # Collapse duplicates in answers by taking the modal label per id (fallback to first).
    def _mode_or_first(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else s.iloc[0]

    answers_grouped = (
        answers_simple.dropna(subset=[target_col])
        .groupby(id_col, as_index=False)[target_col]
        .agg(_mode_or_first)
    )

    merged = submission_simple.merge(
        answers_grouped, on=id_col, how="inner", suffixes=("_pred", "_true")
    )
    if len(merged) == 0:
        raise InvalidSubmissionError(
            f"Submission and answers have no overlapping `{id_col}` values."
        )

    # Rebuild aligned dfs for the shared helper.
    submission_aligned = merged[[id_col, f"{target_col}_pred"]].rename(
        columns={f"{target_col}_pred": target_col}
    )
    answers_aligned = merged[[id_col, f"{target_col}_true"]].rename(
        columns={f"{target_col}_true": target_col}
    )
    
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission_aligned,
        answers=answers_aligned,
        target_column=target_col,
        id_column=id_col,
    )
    return accuracy_score(**accuracy_inputs)
