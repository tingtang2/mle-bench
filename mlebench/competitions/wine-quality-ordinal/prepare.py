from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test[["Id"]].copy()
    # Use the median quality from training data as a sample
    sample_quality = int(old_train["quality"].median())
    sample_submission["quality"] = sample_quality
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(["quality"], axis=1, inplace=True)
    new_test.to_csv(public / "test.csv", index=False)

    # Checks
    assert "quality" not in new_test.columns, "Public test set should not have 'quality' column"
    assert "quality" in new_train.columns, "Public train set should have 'quality' column"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
    assert sample_submission.columns.to_list() == [
        "Id",
        "quality",
    ], "Sample submission columns should only be 'Id' and 'quality'"
    assert len(sample_submission) == len(new_test), "Sample submission length should match test set"

