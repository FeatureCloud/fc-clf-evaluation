# Classification Evaluation FeatureCloud App

## Description
A Classification Evaluation FeautureCloud app, allowing to evaluate your trained models with various classification metrics (e.g. Accuracy).

## Input
- test.csv containing the actual test dataset
- pred.csv containing the predictions oof the model on the test dataset

## Output
- score.csv containing various evaluation metrics

## Workflows
Can be combined with the following apps:
- Pre: Various classification apps (e.g. Random Forest, Logistic Regression, ...)

## Config
Use the config file to customize the evaluation. Just upload it together with your training data as `config.yml`
```
fc_classification_evaluation:
  input:
    y_true: "test.csv"
    y_pred: "pred.csv"
  format:
    sep: ","
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
```
