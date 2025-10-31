# Task

Predict whether a client will subscribe to a bank term deposit.

# Metric

ROC AUC between the predicted value and the observed target.

# Submission Format

For each id in the test set, you must predict the probability of the binary target y. The file should contain a header and have the following format:

```
id,y
750000,0.5
750001,0.5
750002,0.5
..
..
```

# Dataset

The dataset for this competition includes both train and test data with features related to bank marketing campaigns.

