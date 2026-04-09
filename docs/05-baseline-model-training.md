# Baseline Model Training

## Why a baseline model stage was required

A data science assignment should not begin with the most complex model.

The baseline stage exists to answer three questions:

1. can the target be predicted at all
2. how much value comes from a simple interpretable model
3. is accuracy misleading because of class imbalance

## Baseline models used

The baseline training script is `scripts/03_train_baseline_models.py`.

Models:

- `DummyClassifier`
- `LogisticRegression`
- `RandomForestClassifier`

## Why these models were chosen

- The dummy model is a no-skill reference point.
- Logistic regression is interpretable and standard for tabular classification.
- Random forest checks whether non-linear structure helps without going straight to a boosted model.

## Why the evaluation setup looks this way

- **Stratified 80/20 split**: preserves the diabetes class balance
- **5-fold cross-validation**: reduces the risk of a misleading single split
- **ROC-AUC**: measures ranking ability
- **PR-AUC**: important because diabetes is the minority class
- **F1-score**: balances precision and recall
- **Accuracy**: included, but not trusted alone

## Why logistic regression stays important

Even after improvement, logistic regression remains the best explanatory model.

It is the model that best supports:

- coefficient interpretation
- odds-ratio discussion
- feature-level explanation in the written report

## Baseline artifacts

The baseline stage writes to `outputs/stages/03_baseline_models/`.

