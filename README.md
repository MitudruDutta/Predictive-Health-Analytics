# Predictive-Health-Analytics

This repository contains a staged data science workflow for the assignment topic:

> Predictive Health: Lifestyle and Chronic Disease

## Project structure

- `predictive_health/`: reusable ETL, screening, feature engineering, and modeling logic
- `scripts/`: runnable stage-by-stage scripts
- `docs/`: written documentation for the assignment
- `data/`: local source data and dataset notes

## Recommended run order

Activate the environment first:

```bash
source ~/python/bin/activate
```

Then run the project in order:

```bash
python scripts/01_target_screening.py
python scripts/02_etl_diabetes_datasets.py
python scripts/03_train_baseline_models.py
python scripts/04_train_richer_catboost.py
python scripts/train_diabetes_model.py
```

## Documentation map

- `docs/00-workflow-map.md`
- `docs/01-problem-statement-and-topic-choice.md`
- `docs/02-dataset-audit-and-screening.md`
- `docs/03-etl-and-cleaning.md`
- `docs/04-feature-engineering.md`
- `docs/05-baseline-model-training.md`
- `docs/06-model-improvement-and-final-score.md`
- `docs/predictive-health-analysis.md`
