import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Load the original data
    old_train = pd.read_csv(raw / "train.csv")
    old_test = pd.read_csv(raw / "test.csv")

    # Calculate test ratio from original split
    test_ratio = len(old_test) / (len(old_train) + len(old_test))

    # Get the target column name (assuming it's 'y' based on the description)
    target_col = "y"
    id_col = "id"

    # Verify the target column exists in training data
    assert target_col in old_train.columns, f"Target column '{target_col}' not found in train.csv"
    assert id_col in old_train.columns, f"ID column '{id_col}' not found in train.csv"

    # Create new train/test split from the old train data
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Remove target column from new test set (what the agent sees)
    new_test_without_labels = new_test.drop(columns=[target_col])

    # Save public data (what the agent can access)
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Create answers (ground truth for grading)
    answers = new_test[[id_col, target_col]].copy()
    answers.to_csv(private / "test.csv", index=False)

    # Create sample submission (template for agents)
    sample_submission = new_test[[id_col]].copy()
    sample_submission[target_col] = 0.5  # Default probability of 0.5
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    # Validation checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} total samples in new_train ({len(new_train)}) and new_test ({len(new_test)})"
    
    assert len(new_test) == len(
        new_test_without_labels
    ), f"Expected new_test ({len(new_test)}) to have the same length as new_test_without_labels ({len(new_test_without_labels)})"
    
    assert len(answers) == len(
        new_test
    ), f"Expected answers ({len(answers)}) to have the same length as new_test ({len(new_test)})"
    
    assert len(sample_submission) == len(
        new_test
    ), f"Expected sample_submission ({len(sample_submission)}) to have the same length as new_test ({len(new_test)})"
    
    assert set(answers.columns) == {id_col, target_col}, (
        f"Answers must have '{id_col}' and '{target_col}' columns"
    )
    
    assert set(sample_submission.columns) == {id_col, target_col}, (
        f"Sample submission must have '{id_col}' and '{target_col}' columns"
    )
    
    # Verify no label leakage in public test data
    assert target_col not in new_test_without_labels.columns, (
        f"Target column '{target_col}' should not be in the public test data"
    )

