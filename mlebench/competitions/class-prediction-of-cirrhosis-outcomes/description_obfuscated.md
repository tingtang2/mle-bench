# Task

Predict the outcomes of patients with cirrhosis using a multi-class approach.

# Metric

Multi-class logarithmic loss.

The metric is calculated as:

log loss = -1/N * sum(sum(y_ij * log(p_ij))) for i in N and j in M

where N is the number of rows in the test set, M is the number of outcomes (i.e., 3), log is the natural logarithm, y_ij is 1 if row i has the ground truth label j and 0 otherwise, and p_ij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(1e-15, min(p_ij, 1-1e-15)).

# Submission Format

For each id row in the test set, you must predict probabilities of the three outcomes Status_C, Status_CL, and Status_D. The file should contain a header and have the following format:

```
id,Status_C,Status_CL,Status_D
7905,0.628084,0.034788,0.337128
7906,0.628084,0.034788,0.337128
7907,0.628084,0.034788,0.337128
etc.
```

# Dataset

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Cirrhosis Patient Survival Prediction dataset. Feature distributions are close to, but not exactly the same, as the original.

## Data fields

train.csv - the training dataset; Status is the categorical target; C (censored) indicates the patient was alive at N_Days, CL indicates the patient was alive at N_Days due to liver a transplant, and D indicates the patient was deceased at N_Days.

test.csv - the test dataset; your objective is to predict the probability of each of the three Status values, e.g., Status_C, Status_CL, Status_D.

sample_submission.csv - a sample submission file in the correct format

