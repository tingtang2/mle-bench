# Task

Predict values for synthetic data.

## Description

# Metric

Quadratic weighted kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.

The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of Ids i (actual) that received a predicted value j. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values:

w_i_j = (i - j)^2 / (N - 1)^2

An N-by-N histogram matrix of expected outcomes, E, is calculated assuming that there is no correlation between values. This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 

```
kappa = 1- (sum(sum(w_i_j * O_i_j)) / sum(sum(w_i_j * E_i_j)))
```

# Submission Format

For each `Id` in the test set, you must predict the value for the target. The file should contain a header and have the following format:

```
Id,target
2056,5
2057,7
2058,3
etc.
```

# Dataset 

- **train.csv** - the training dataset; target is the ordinal target
- **test.csv** - the test dataset; your objective is to predict the target for each row
- **sample_submission.csv** - a sample submission file in the correct format

