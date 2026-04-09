# Feature Engineering

## Why feature engineering was needed

Raw NHIS codes are not the best form for modeling or explanation.

- Grouped categories are easier to interpret than code numbers.
- Feature engineering reduces noise from overly granular survey encodings.
- The assignment requires explaining why variables were transformed, not only using them.

## Baseline feature set

The compact baseline feature set was built first because it is easy to justify and explain.

Main engineered features:

- `age`
- `sex`
- `education_group`
- `poverty_group`
- `bmi_group`
- `ever_smoked_100_cigs`
- `ever_used_ecig`
- `alcohol_group`
- `activity_level`
- `sleep_group`
- `food_security`

## Richer feature set

A second richer feature set was created only after the baseline was stable.

Additional richer features:

- `smoking_status`
- `activity_guideline_combo_code`
- `region`
- `urban_rural`
- `marital_status`

## Why the richer feature set was added

- to test whether the baseline was leaving useful signal unused
- to check whether a stronger tabular model could gain from richer categorical structure
- to improve score without using leakage-prone diabetes follow-up variables

## Why leakage prevention matters here

Variables that are direct consequences of being diagnosed with diabetes should not be used as predictors.

Examples of variables that should stay out:

- diabetes medication variables
- diabetes treatment variables
- diabetes diagnosis-age variables
- other post-diagnosis follow-up fields

## Feature engineering outputs

The feature engineering stage is saved through the ETL outputs and reused by both baseline and richer-model training scripts.

