# Overview

## Description

Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Your objective is to predict whether a person is an Introvert or Extrovert, given their social behavior and personality traits.

## Evaluation

Submissions are evaluated using Accuracy Score between the predicted value and the observed target.

### Submission File

For each id in the test set, you must predict the target Personality. The file should contain a header and have the following format:

```
id,Personality
18524,Extrovert
18525,Introvert
18526,Introvert
etc.
```

### Citation

Walter Reade and Elizabeth Park. Predict the Introverts from the Extroverts. https://kaggle.com/competitions/playground-series-s5e7, 2025. Kaggle.


# Data

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Extrovert vs. Introvert Behavior dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Note - This is a relatively small dataset, so a good one to use for comparing different modeling approaches, making visualizations, etc.

## Data fields

train.csv - the training dataset; Personality is the categorical target

test.csv - the test dataset; your objective is to predict the Personality for each row

sample_submission.csv - a sample submission file in the correct format