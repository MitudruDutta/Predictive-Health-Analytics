# ETL And Cleaning

## Why ETL was needed

The raw NHIS file is not modeling-ready.

- Survey codes include non-response values that are not real measurements.
- Some fields need regrouping before they become analytically useful.
- The project needs a smaller, focused modeling table instead of all 630 raw columns.

## What ETL means in this project

Here ETL stands for:

1. **Extract** the relevant raw NHIS columns
2. **Transform** survey codes into valid modeling variables
3. **Load** engineered diabetes datasets for downstream modeling

## How ETL was implemented

The ETL stage is implemented in:

- `predictive_health/io.py`
- `predictive_health/features.py`
- `predictive_health/etl.py`
- `scripts/02_etl_diabetes_datasets.py`

## Cleaning rules

Survey special codes were converted to missing values before modeling. Examples:

- `SEX_A`: `7`, `9`
- `EDUCP_A`: `97`, `99`
- `BMICATD_A`: `9`
- `SMKEV_A`, `SMKNOW_A`, `ECIGEV_A`, `DIBEV_A`: `7`, `8`, `9`
- `DRKSTAT_A`: `10`
- `PA18_02R_A`, `PA18_05R_A`: `8`
- `SLPHOURS_A`: `97`, `98`, `99`
- `FDSCAT3_A`: `8`

## Why rows were only dropped when the target was invalid

Rows were kept unless the diabetes target itself was missing.

That was done because:

- predictor missingness is low enough to handle with imputation
- deleting rows too early would waste data
- class imbalance makes data retention important

## ETL outputs

The ETL stage writes:

- `outputs/stages/02_etl/baseline_diabetes_dataset.csv`
- `outputs/stages/02_etl/richer_diabetes_dataset.csv`
- missingness summaries for both datasets

