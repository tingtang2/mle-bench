# Overview

## Description

Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Your goal is to predict whether a client will subscribe to a bank term deposit.

### Evaluation

Submissions are evaluated using ROC AUC between the predicted value and the observed target.

### Submission File

For each id in the test set, you must predict the probability of the binary target y. The file should contain a header and have the following format:

```
id,y
750000,0.5
750001,0.5
750002,0.5
..
..
```

### Citation

Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset. https://kaggle.com/competitions/playground-series-s5e8, 2025. Kaggle.

# Data

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Bank Marketing Dataset dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

