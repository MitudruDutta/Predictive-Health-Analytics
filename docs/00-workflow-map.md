# Workflow Map

This project is now split into separate stages so the assignment can be explained from the beginning instead of only from the final model.

## Execution order

1. `scripts/01_target_screening.py`
   - screens candidate chronic-disease targets
   - checks whether the topic is feasible with this dataset
2. `scripts/02_etl_diabetes_datasets.py`
   - performs ETL and builds the diabetes modeling datasets
   - saves both the compact baseline dataset and the richer dataset
3. `scripts/03_train_baseline_models.py`
   - trains the interpretable baseline models
   - saves metrics and logistic diagnostics
4. `scripts/04_train_richer_catboost.py`
   - trains the stronger richer-feature model
   - saves CatBoost diagnostics and the improved score
5. `scripts/train_diabetes_model.py`
   - runs the full final pipeline and writes the consolidated final artifacts used in the report

## Why this structure was created

- It makes the assignment easier to explain step by step.
- It separates ETL from modeling, which is the right data science workflow.
- It lets us prove that the target was chosen deliberately and not arbitrarily.
- It preserves both the interpretable baseline and the higher-scoring model.

## Core package

The reusable logic lives in the `predictive_health/` package:

- `config.py`: paths, column lists, and special-code mappings
- `io.py`: loading, saving, and output-directory helpers
- `screening.py`: target screening and feature-set comparison
- `features.py`: grouped feature transforms
- `etl.py`: engineered dataset construction
- `modeling.py`: baseline and CatBoost training plus diagnostics

## Main documentation files

- `docs/01-problem-statement-and-topic-choice.md`
- `docs/02-dataset-audit-and-screening.md`
- `docs/03-etl-and-cleaning.md`
- `docs/04-feature-engineering.md`
- `docs/05-baseline-model-training.md`
- `docs/06-model-improvement-and-final-score.md`
- `docs/predictive-health-analysis.md`

