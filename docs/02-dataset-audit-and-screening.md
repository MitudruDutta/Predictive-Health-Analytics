# Dataset Audit And Screening

## Dataset used

This project uses the local file `data/adult24.csv`, which is the 2024 NHIS Sample Adult public-use dataset.

- Rows: `32,629`
- Columns: `630`
- Data type: cross-sectional health survey
- Important note: the dataset uses coded missing values such as `7`, `8`, `9`, `97`, `98`, and `99`

## Why dataset auditing was necessary

Auditing comes before ETL and modeling because a project should first confirm:

- whether the dataset actually fits the topic
- whether valid target variables exist
- whether missingness is manageable
- whether a meaningful predictive question can be framed from the data

## What was checked

The screening script `scripts/01_target_screening.py` checks:

1. selected-feature missingness after recoding survey missing values
2. candidate chronic-disease targets
3. feature-set comparisons for diabetes and hypertension

## Candidate targets screened

- `DIBEV_A`: diabetes
- `PREDIB_A`: prediabetes
- `HYPEV_A`: hypertension
- `CHDEV_A`: coronary heart disease
- `PHSTAT_A`: fair or poor health versus better health

## Why the screening stage matters

It prevents a weak assignment setup. Instead of assuming the topic is feasible, we tested whether the target is actually learnable from the available predictors.

That is why this stage exists before ETL and training.

## Saved screening artifacts

The screening outputs are saved under `outputs/stages/01_screening/`:

- `selected_feature_missingness.csv`
- `target_screening.csv`
- `feature_set_comparison.csv`
- `summary.txt`

