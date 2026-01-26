import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Find id and target columns
    id_col = None
    target_col = None
    
    for col in ['id', 'Id', 'ID', 'date', 'timestamp', 'time', 'datetime', 'index']:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]
    
    for col in ['class', 'demand', 'electricity_demand', 'consumption', 'value', 'target', 'y', 'label']:
        if col in answers.columns:
            target_col = col
            break
    if target_col is None:
        target_col = [c for c in answers.columns if c != id_col][-1]

    # Some prepared splits store labels as strings like "b'UP'" / "b'DOWN'".
    # Normalize both y_true and y_pred to comparable canonical strings.
    def _normalize_labels(df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            return
        s = df[col]
        # Ensure string dtype (handles bytes-like, objects, etc.)
        s = s.astype(str).str.strip()
        # Convert "b'UP'" -> "UP"
        s = s.str.replace(r"^b'(.+)'$", r"\1", regex=True)
        s = s.str.replace(r'^b"(.+)"$', r"\1", regex=True)
        df[col] = s

    _normalize_labels(submission, target_col)
    _normalize_labels(answers, target_col)
    
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column=target_col, id_column=id_col
    )
    return accuracy_score(**accuracy_inputs)
