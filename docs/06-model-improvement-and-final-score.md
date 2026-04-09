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

## What exact steps were performed to improve the score

The improvement process was done in three stages.

1. Build a stronger feature set
   - Added `smoking_status`
   - Added `activity_guideline_combo_code`
   - Added `region`
   - Added `urban_rural`
   - Added `marital_status`
   - Why: these add behavioral and contextual signal without using diabetes follow-up leakage variables
2. Move from the interpretable baseline to a stronger tabular model
   - Switched from only logistic regression and random forest to `CatBoostClassifier`
   - Why: CatBoost handles grouped categorical variables well
3. Run a focused CatBoost tuning sweep
   - Tested several depth, learning-rate, iteration, and regularization settings
   - Why: to check whether the saved richer CatBoost model was under-tuned

## What happened when we tried to improve it further

The tuning sweep showed that only a very small additional gain is still available.

- Best tuned ROC-AUC: `0.7793` using a shallower-but-longer CatBoost setup
- Best saved PR-AUC remains the current richer CatBoost model at `0.2891`

This means:

- yes, the score can still move a little
- no, there is not a large untapped improvement from simple tuning alone
- the current richer CatBoost model is already close to the practical limit of this feature set and target definition

## Should we replace the current saved model with the tuned one?

Not necessarily.

The tuned variant improves ROC-AUC slightly, but it reduces PR-AUC slightly. For this assignment, that is not a clear universal win.

So the honest conclusion is:

- if ROC-AUC is the main target, a tiny improvement is possible
- if PR-AUC and balanced practical usefulness matter, the current saved richer CatBoost result is still the cleaner final choice

## Why the improvement is still meaningful

The gain is modest, not dramatic. That matters.

- It shows the baseline was already strong.
- It shows the richer model found additional signal rather than exploiting leakage.
- It keeps the final conclusion realistic: this is a screening model, not a diagnostic system.

## Final saved artifacts

The consolidated final outputs remain in `outputs/diabetes_model/`.

The tuning evidence is saved separately in:

- `outputs/stages/05_tuning/catboost_tuning_results.csv`
- `outputs/stages/05_tuning/summary.txt`
