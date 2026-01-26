from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'user_id', 'customer_id']:
        if col in old_train.columns:
            id_col = col
            break
    if id_col is None:
        id_col = old_train.columns[0]
    
    # Find prediction/target column (for recommendations, might be 'prediction' or similar)
    pred_col = None
    for col in ['prediction', 'target', 'items', 'recommendations']:
        if col in old_train.columns:
            pred_col = col
            break
    if pred_col is None:
        # Use the last column
        pred_col = [c for c in old_train.columns if c != id_col][-1]

    # Create sample submission
    sample_submission = new_test[[id_col]].copy()
    # Default prediction: empty or a placeholder
    sample_submission[pred_col] = ""
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    # Remove target column from test
    new_test_without_target = new_test.drop(pred_col, axis=1)
    new_test_without_target.to_csv(public / "test.csv", index=False)
