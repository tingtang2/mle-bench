# Task

Predict values for synthetic data.

## Description

# Metric

Area under the ROC curve between the predicted probability and the observed target.

# Submission Format

For each `id` in the test set, you must predict a probability for the target. The file should contain a header and have the following format:

```
id,target
2190,0.5
2191,0.1
2192,0.9
etc.
```

# Dataset 

- **train.csv** - the training dataset; target is the binary target
- **test.csv** - the test dataset; your objective is to predict the probability of the target for each row
- **sample_submission.csv** - a sample submission file in the correct format

