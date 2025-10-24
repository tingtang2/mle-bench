from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# The two personality classes
CLASSES = ["Introvert", "Extrovert"]


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Load the original data
    old_train = pd.read_csv(raw / "train.csv")
    
    # Check if test.csv exists to calculate test ratio
    test_ratio = 0.1  # default
    if (raw / "test.csv").exists():
        old_test = pd.read_csv(raw / "test.csv")
        test_ratio = len(old_test) / (len(old_train) + len(old_test))

    # Get the target column name
    target_col = "Personality"
    id_col = "id"

    # Verify the target column exists in training data
    assert target_col in old_train.columns, f"Target column '{target_col}' not found in train.csv"
    assert id_col in old_train.columns, f"ID column '{id_col}' not found in train.csv"
    
    # Verify Personality has the expected values (Introvert, Extrovert)
    unique_personalities = set(old_train[target_col].unique())
    expected_personalities = set(CLASSES)
    assert unique_personalities.issubset(expected_personalities), (
        f"Expected Personality values to be subset of {expected_personalities}, got {unique_personalities}"
    )

    # Create new train/test split from the old train data
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Remove target column from new test set (what the agent sees)
    new_test_without_labels = new_test.drop(columns=[target_col])

    # Create answers (ground truth for grading)
    answers = new_test[[id_col, target_col]].copy()

    # Save public data (what the agent can access)
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Save private data (ground truth for grading)
    answers.to_csv(private / "test.csv", index=False)

    # Create sample submission (template for agents)
    # Set all predictions to "Introvert" as a baseline
    sample_submission = new_test[[id_col]].copy()
    sample_submission[target_col] = "Introvert"
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

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
    
    # Verify that sample submission has proper shape
    assert sample_submission.shape == (len(new_test), 2), (
        f"Sample submission should have shape ({len(new_test)}, 2) with columns [id, Personality]"
    )
    
    print(f"Prepared dataset:")
    print(f"  Training samples: {len(new_train)}")
    print(f"  Test samples: {len(new_test)}")
    print(f"  Classes: {CLASSES}")

