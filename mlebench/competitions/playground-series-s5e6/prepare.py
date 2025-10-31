from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test.copy()
    # Use the first fertilizer name from train set as a sample (up to 3 values space-delimited)
    # Get a common fertilizer name from training data
    sample_fertilizer = old_train["Fertilizer Name"].value_counts().index[0]
    sample_submission["Fertilizer Name"] = sample_fertilizer
    sample_submission.drop(
        sample_submission.columns.difference(["id", "Fertilizer Name"]), axis=1, inplace=True
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(["Fertilizer Name"], axis=1, inplace=True)
    new_test.to_csv(public / "test.csv", index=False)

    # Checks
    assert "Fertilizer Name" not in new_test.columns, "Public test set should not have 'Fertilizer Name' column"
    assert "Fertilizer Name" in new_train.columns, "Public train set should have 'Fertilizer Name' column"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
    assert sample_submission.columns.to_list() == [
        "id",
        "Fertilizer Name",
    ], "Sample submission columns should only be 'id' and 'Fertilizer Name'"
    assert len(sample_submission) == len(new_test), "Sample submission length should match test set"

