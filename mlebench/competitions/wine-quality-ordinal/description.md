# Overview

## Description

Wine Quality Prediction: Ordinal Regression Challenge is a competition focused on predicting wine quality ratings using ordinal regression.

## Evaluation

### Metric

Submissions are evaluated using Quadratic Weighted Kappa (QWK). This metric assesses the agreement between the predicted and actual wine quality ratings, accounting for the ordinal nature of the target variable. QWK assigns greater penalties to predictions that are further from the true value, making it particularly suitable for ordinal classification tasks.

### Submission Format

The submission format is a csv file with predicted quality ratings for each test sample.
