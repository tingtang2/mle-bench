from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot

# The three status classes (raw values in data)
RAW_CLASSES = ["C", "CL", "D"]
# The three status classes (for submission format)
CLASSES = ["Status_C", "Status_CL", "Status_D"]


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
    target_col = "Status"  # As per description: Status is the categorical target
    id_col = "id"

    # Verify the target column exists in training data
    assert target_col in old_train.columns, f"Target column '{target_col}' not found in train.csv"
    assert id_col in old_train.columns, f"ID column '{id_col}' not found in train.csv"
    
    # Verify Status has the expected values (C, CL, D)
    unique_statuses = set(old_train[target_col].unique())
    expected_statuses = set(RAW_CLASSES)
    assert unique_statuses.issubset(expected_statuses), (
        f"Expected Status values to be subset of {expected_statuses}, got {unique_statuses}"
    )

    # Create new train/test split from the old train data
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    # Remove target column from new test set (what the agent sees)
    new_test_without_labels = new_test.drop(columns=[target_col])

    # Convert Status to one-hot encoding for answers
    # Status values C, CL, D become Status_C, Status_CL, Status_D
    new_test_answers = new_test[[id_col, target_col]].copy()
    new_test_answers = df_to_one_hot(
        new_test_answers, 
        id_column=id_col, 
        target_column=target_col, 
        classes=RAW_CLASSES  # Use raw class values that exist in the data
    )
    
    # Rename columns from C, CL, D to Status_C, Status_CL, Status_D
    column_mapping = {raw: prefixed for raw, prefixed in zip(RAW_CLASSES, CLASSES)}
    new_test_answers = new_test_answers.rename(columns=column_mapping)

    # Save public data (what the agent can access)
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Save private data (ground truth for grading)
    new_test_answers.to_csv(private / "test.csv", index=False)

    # Create sample submission (template for agents)
    # Each status class gets equal probability (1/3)
    sample_submission = new_test[[id_col]].copy()
    for cls in CLASSES:
        sample_submission[cls] = 1.0 / len(CLASSES)
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    # Validation checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} total samples in new_train ({len(new_train)}) and new_test ({len(new_test)})"
    
    assert len(new_test) == len(
        new_test_without_labels
    ), f"Expected new_test ({len(new_test)}) to have the same length as new_test_without_labels ({len(new_test_without_labels)})"
    
    assert len(new_test_answers) == len(
        new_test
    ), f"Expected new_test_answers ({len(new_test_answers)}) to have the same length as new_test ({len(new_test)})"
    
    assert len(sample_submission) == len(
        new_test
    ), f"Expected sample_submission ({len(sample_submission)}) to have the same length as new_test ({len(new_test)})"
    
    assert set(new_test_answers.columns) == {id_col} | set(CLASSES), (
        f"Answers must have '{id_col}' column and class columns: {CLASSES}"
    )
    
    assert set(sample_submission.columns) == {id_col} | set(CLASSES), (
        f"Sample submission must have '{id_col}' column and class columns: {CLASSES}"
    )
    
    # Verify no label leakage in public test data
    assert target_col not in new_test_without_labels.columns, (
        f"Target column '{target_col}' should not be in the public test data"
    )
    
    # Verify that sample submission has proper probabilities
    assert sample_submission[CLASSES].shape == (len(new_test), len(CLASSES)), (
        f"Sample submission should have shape ({len(new_test)}, {len(CLASSES)})"
    )

