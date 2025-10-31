# Overview

## Description

### Welcome to the 2025 Kaggle Playground Series!

We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Your objective is to select the best fertilizer for different weather, soil conditions and crops.

## Evaluation

Submissions are evaluated according to the Mean Average Precision @ 3 (MAP@3):

MAP@3 = 1/U * sum(sum P(k) * rel(k)) for u in U and k in 1 to min(n,3)

where U is the number of observations, P(k) is the precision at cutoff k, n is the number predictions per observation, and rel(k) is an indicator function equaling 1 if the item at rank k is a relevant (correct) label, zero otherwise.

Once a correct label has been scored for an observation, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation. For example, if the correct label is A for an observation, the following predictions all score an average precision of 1.0:

```
[A, B, C, D, E]
[A, A, A, A, A]
[A, B, A, C, A]
```

### Submission File

For each `id` in the test set, you may predict up to 3 Fertilizer Name values, with the predictions space delimited. The file should contain a header and have the following format:

```
id,Fertilizer Name
750000,14-35-14 10-26-26 Urea
750001,14-35-14 10-26-26 Urea
...
```

## Timeline

- **Start Date** - May 31, 2025
- **Final Submission Deadline** - June 30, 2025

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted.

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Fertilizer prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

### Files

- **train.csv** - the training dataset; `Fertilizer Name` is the categorical target
- **test.csv** - the test dataset; your objective is to predict the `Fertilizer Name` for each row, up to three values, space delimited
- **sample_submission.csv** - a sample submission file in the correct format
