from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    target_col = None
    for col in ['target', 'y', 'label']:
        if col in old_train.columns:
            target_col = col
            break
    if target_col is None:
        target_col = old_train.columns[-1]

    id_col = None
    for col in ['id', 'Id', 'ID', 'sample_id']:
        if col in old_train.columns:
            id_col = col
            break
    if id_col is None:
        id_col = old_train.columns[0]

    sample_submission = new_test[[id_col]].copy()
    sample_submission[target_col] = 0.5
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    new_test.to_csv(private / "test.csv", index=False)

    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(columns=[target_col], errors="ignore").to_csv(public / "test.csv", index=False)
