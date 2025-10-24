from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# The five polymer properties
PROPERTIES = ["density", "Tc", "Tg", "Rg", "FFV"]


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

    # Verify required columns exist
    assert "id" in old_train.columns, "ID column 'id' not found in train.csv"
    
    # Verify at least some of the property columns exist
    existing_properties = [prop for prop in PROPERTIES if prop in old_train.columns]
    assert len(existing_properties) > 0, (
        f"None of the expected property columns {PROPERTIES} found in train.csv"
    )

    # Create new train/test split from the old train data
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Remove property columns from new test set (what the agent sees)
    columns_to_drop = [prop for prop in PROPERTIES if prop in new_test.columns]
    new_test_without_labels = new_test.drop(columns=columns_to_drop)

    # Create answers (ground truth for grading) - keep only id and property columns
    answer_columns = ["id"] + [prop for prop in PROPERTIES if prop in new_test.columns]
    answers = new_test[answer_columns].copy()

    # Save public data (what the agent can access)
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Save private data (ground truth for grading)
    answers.to_csv(private / "test.csv", index=False)

    # Create sample submission (template for agents)
    # Set all predictions to 1.0 as a baseline
    sample_submission = new_test[["id"]].copy()
    for prop in PROPERTIES:
        sample_submission[prop] = 1.0
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
    
    # Check that answers has id and property columns
    expected_answer_cols = {"id"} | set([prop for prop in PROPERTIES if prop in old_train.columns])
    assert set(answers.columns) == expected_answer_cols, (
        f"Answers must have 'id' column and property columns. "
        f"Expected: {expected_answer_cols}, Got: {set(answers.columns)}"
    )
    
    # Check that sample submission has id and all property columns
    expected_submission_cols = {"id"} | set(PROPERTIES)
    assert set(sample_submission.columns) == expected_submission_cols, (
        f"Sample submission must have 'id' column and all property columns: {PROPERTIES}. "
        f"Got: {set(sample_submission.columns)}"
    )
    
    # Verify no label leakage in public test data
    for prop in PROPERTIES:
        if prop in old_train.columns:
            assert prop not in new_test_without_labels.columns, (
                f"Property column '{prop}' should not be in the public test data"
            )
    
    # Verify that sample submission has proper shape
    assert sample_submission.shape == (len(new_test), len(PROPERTIES) + 1), (
        f"Sample submission should have shape ({len(new_test)}, {len(PROPERTIES) + 1})"
    )
    
    print(f"Prepared dataset:")
    print(f"  Training samples: {len(new_train)}")
    print(f"  Test samples: {len(new_test)}")
    print(f"  Properties: {existing_properties}")

