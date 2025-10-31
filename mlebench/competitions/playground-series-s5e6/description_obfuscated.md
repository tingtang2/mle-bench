# Task

Predict values for synthetic data.

## Description

# Metric

Mean Average Precision @ 3 (MAP@3):

MAP@3 = 1/U * sum(sum P(k) * rel(k)) for u in U and k in 1 to min(n,3)

where U is the number of observations, P(k) is the precision at cutoff k, n is the number predictions per observation, and rel(k) is an indicator function equaling 1 if the item at rank k is a relevant (correct) label, zero otherwise.

Once a correct label has been scored for an observation, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation.

# Submission Format

For each `id` in the test set, you may predict up to 3 target values, with the predictions space delimited. The file should contain a header and have the following format:

```
id,Target Name
750000,Value1 Value2 Value3
750001,Value1 Value2 Value3
etc.
```

# Dataset 

- **train.csv** - the training dataset; target is the categorical target
- **test.csv** - the test dataset; your objective is to predict the target for each row, up to three values, space delimited
- **sample_submission.csv** - a sample submission file in the correct format

