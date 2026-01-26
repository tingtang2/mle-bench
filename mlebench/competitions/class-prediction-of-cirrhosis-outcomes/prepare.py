from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Find target columns (Status_C, Status_CL, Status_D)
    target_cols = [col for col in old_train.columns if col.startswith('Status_')]
    
    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'sample_id']:
        if col in old_train.columns:
            id_col = col
            break
    if id_col is None:
        id_col = old_train.columns[0]

    # Create sample submission with probabilities
    sample_submission = new_test[[id_col]].copy()
    for col in target_cols:
        sample_submission[col] = 1.0 / len(target_cols)  # Equal probabilities
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(target_cols, axis=1).to_csv(public / "test.csv", index=False)
