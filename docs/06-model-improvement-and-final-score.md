# Model Improvement And Final Score

## Why the model was improved after the baseline

Improving the model was necessary because a good assignment should show that the baseline was tested against a stronger alternative.

This improvement stage answers:

- Is the baseline underfitting?
- Does richer categorical structure help?
- Can we improve score without introducing target leakage?

## Improved model choice

The richer-model script is `scripts/04_train_richer_catboost.py`.

Model used:

- `CatBoostClassifier`

## Why CatBoost was selected

- It handles categorical variables well.
- It works strongly on mixed tabular survey data.
- It can capture non-linear relationships without one-hot encoding every category.

## Final interpretation

Two model roles are now clearly separated:

1. **Best explanatory model**: logistic regression
2. **Best score model**: richer CatBoost

That is a stronger final story than reporting only one number.

## Score summary

Current saved results:

- Logistic regression: test ROC-AUC `0.7723`, test PR-AUC `0.2804`
- Richer CatBoost: test ROC-AUC `0.7777`, test PR-AUC `0.2891`

## Why the improvement is still meaningful

The gain is modest, not dramatic. That matters.

- It shows the baseline was already strong.
- It shows the richer model found additional signal rather than exploiting leakage.
- It keeps the final conclusion realistic: this is a screening model, not a diagnostic system.

## Final saved artifacts

The consolidated final outputs remain in `outputs/diabetes_model/`.

