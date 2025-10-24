# Task

Predict whether a person is an Introvert or Extrovert, given their social behavior and personality traits.

# Metric

Accuracy Score between the predicted value and the observed target.

# Submission Format

For each id in the test set, you must predict the target Personality. The file should contain a header and have the following format:

```
id,Personality
18524,Extrovert
18525,Introvert
18526,Introvert
etc.
```

# Dataset

The dataset contains social behavior and personality trait features to predict whether individuals are Introverts or Extroverts. The data was generated from a deep learning model trained on the Extrovert vs. Introvert Behavior dataset.

## Data fields

train.csv - the training dataset; Personality is the categorical target

test.csv - the test dataset; your objective is to predict the Personality for each row

sample_submission.csv - a sample submission file in the correct format

