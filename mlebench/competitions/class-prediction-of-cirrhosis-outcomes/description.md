# Overview

## Description

Multi-Class Prediction of Cirrhosis Outcomes is a competition focused on predicting the probabilities of three possible outcomes: Status_C, Status_CL, and Status_D for cirrhosis patients.

## Evaluation

### Metric

Submissions are evaluated using multi-class logarithmic loss (log loss). The metric measures the performance of a classification model whose output is a probability value between 0 and 1. It quantifies the difference between the predicted probability distribution and the actual distribution of the target classes. A lower log loss value indicates better model performance.

### Submission Format

The submission format is a csv file with probabilities for each class for each test sample.
