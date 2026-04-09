# Predictive Health: Lifestyle and Chronic Disease

## 1. Why this topic was chosen

This topic is a strong data science assignment topic because it sits at the intersection of public health relevance, interpretable modeling, and actionable predictors.

- Chronic diseases such as hypertension and diabetes are common, costly, and strongly connected to modifiable lifestyle factors such as body weight, smoking, physical activity, sleep, and diet-related conditions.
- A predictive-health framing is practical. If we can estimate disease risk from lifestyle and demographic patterns, the result is easy to explain to a non-technical audience and useful for prevention-oriented decision making.
- The topic also supports the full data science workflow cleanly: problem framing, survey-data cleaning, feature engineering, baseline modeling, model comparison, and interpretation.
- Most importantly, this dataset already contains both sides of the problem:
  lifestyle-related predictors and chronic-disease outcomes.

## 2. Are datasets available for this topic?

Yes. There are multiple public datasets suitable for "Lifestyle and Chronic Disease" analysis.

### Best public sources

1. **NHIS 2024 Sample Adult**
   - Status: available and already added in this repo as `data/adult24.csv`
   - Why it is useful: one file contains self-reported lifestyle, health status, chronic conditions, and demographic variables for adults.
   - Official source: CDC NHIS 2024 documentation page
   - Link: <https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html>

2. **BRFSS 2024**
   - Status: available
   - Why it is useful: very large U.S. behavioral risk factor survey; excellent for chronic disease classification and health-risk modeling.
   - Official source: CDC BRFSS annual data
   - Link: <https://www.cdc.gov/brfss/annual_data/2024/llcp_multiq.html>

3. **NHANES 2021-2023**
   - Status: available
   - Why it is useful: combines questionnaires with examination and laboratory measures, which is useful if the assignment later needs more objective biomarkers.
   - Official source: CDC NHANES data portal
   - Link: <https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023>

### Why we are using the NHIS dataset in this project

The NHIS file is the best starting point for this assignment because it already contains:

- enough rows for modeling: **32,629** adult records
- enough breadth for feature selection: **630** columns
- direct chronic-disease targets such as `HYPEV_A`, `DIBEV_A`, `PREDIB_A`, and `CHDEV_A`
- lifestyle-related variables such as `BMICATD_A`, `SMKEV_A`, `ECIGEV_A`, `DRKSTAT_A`, `PA18_02R_A`, `SLPHOURS_A`, and `FDSCAT3_A`

It is also simpler than NHANES for a first assignment because it does not require merging separate questionnaire, exam, and lab files before analysis.

## 3. Dataset identification

The file in this repository is:

- `data/adult24.csv`
- official 2024 **NHIS Sample Adult** public-use data
- file size in this repo: about **32 MB**
- observations: **32,629**
- columns: **630**

Important context:

- NHIS is a **complex survey**, not a random CSV of independent users.
- The file includes survey weights such as `WTFA_A`.
- The readme warns that importing the CSV with the wrong type inference can misread variables with many missing values. That matters during cleaning.

## 4. Problem statement

### Main research question

Can we use **lifestyle and basic demographic information** from the 2024 NHIS Sample Adult dataset to predict whether an adult has a chronic disease?

### Candidate targets tested

I screened five realistic targets from the dataset:

- `HYPEV_A`: ever told had hypertension
- `DIBEV_A`: ever told had diabetes
- `PREDIB_A`: ever told had prediabetes
- `CHDEV_A`: ever had coronary heart disease
- `PHSTAT_A`: general health status, recoded to fair/poor vs better health

### Target-screening result

Using a quick logistic-regression baseline with a compact lifestyle-plus-demographic feature set, I got the following cross-validated results:

| Target | Positive rate in sample | Weighted positive rate | ROC-AUC | PR-AUC | Initial verdict |
|---|---:|---:|---:|---:|---|
| Hypertension (`HYPEV_A`) | 37.64% | 31.85% | 0.799 | 0.669 | Strong overall target |
| Diabetes (`DIBEV_A`) | 11.38% | 9.99% | 0.783 | 0.296 | Strong, lifestyle-relevant target |
| Prediabetes (`PREDIB_A`) | 18.64% | 16.70% | 0.743 | 0.366 | Usable secondary target |
| Coronary heart disease (`CHDEV_A`) | 6.49% | 5.01% | 0.826 | 0.217 | Predictable but rare and less ideal for first model |
| Fair/Poor health (`PHSTAT_A`) | 16.42% | 14.84% | 0.788 | 0.421 | Useful but not disease-specific |

### Final choice for the assignment

For this assignment, **diagnosed diabetes (`DIBEV_A`) is the best primary target**, and **hypertension (`HYPEV_A`) is the best secondary benchmark**.

### Why diabetes is the better primary target

- It matches the topic better. Diabetes is a classic chronic disease with strong lifestyle links.
- The model is good enough to justify prediction: **ROC-AUC about 0.783**.
- Lifestyle variables alone still carry meaningful signal. That is important because the assignment is about **lifestyle and chronic disease**, not only age-based risk ranking.
- Diabetes is easier to explain in a report because variables such as BMI, smoking, physical activity, food insecurity, and sleep have a direct public-health interpretation.

### Why hypertension is not the primary target, even though it scores slightly higher

Hypertension is the easiest target overall, but much of that lift comes from **age and demographics**. For a lifestyle-centered assignment, diabetes is the cleaner story.

## 5. What the dataset tells us already

## 5.1 Selected-feature missingness after recoding survey missing values

For the first-pass modeling feature set, missingness is manageable:

| Variable | Missing rate after recoding |
|---|---:|
| `FDSCAT3_A` | 4.14% |
| `PA18_02R_A` | 3.66% |
| `SLPHOURS_A` | 3.45% |
| `DRKSTAT_A` | 2.69% |
| `ECIGEV_A` | 2.27% |
| `SMKEV_A` | 2.26% |
| `BMICATD_A` | 1.84% |
| `EDUCP_A` | 0.45% |
| `SEX_A` | 0.02% |
| `AGEP_A`, `POVRATTC_A` | 0.00% |

### Why this matters

This is good news for the project. It means we do **not** need aggressive row deletion at the start. Simple imputation is enough for a strong baseline.

## 5.2 Evidence that the file fits the topic

The dataset contains variables that directly connect lifestyle and chronic disease:

- body mass index category: `BMICATD_A`
- smoking history: `SMKEV_A`
- e-cigarette use: `ECIGEV_A`
- alcohol status: `DRKSTAT_A`
- physical activity recodes: `PA18_02R_A`, `PA18_05R_A`
- sleep duration: `SLPHOURS_A`
- food security status: `FDSCAT3_A`
- chronic disease outcomes: `DIBEV_A`, `HYPEV_A`, `CHDEV_A`, `PREDIB_A`

That means we do not have to invent a proxy target. The target already exists in the data.

## 5.3 Key observed patterns

### Diabetes rises sharply with BMI

From the local analysis:

| BMI group | Diabetes rate |
|---|---:|
| Underweight | 3.61% |
| Healthy weight | 6.08% |
| Overweight | 10.35% |
| Class 1 obesity | 16.42% |
| Class 2 obesity | 17.85% |
| Class 3 obesity | 22.41% |

This is one of the clearest reasons diabetes is a strong target for this assignment.

### Hypertension also rises sharply with BMI

| BMI group | Hypertension rate |
|---|---:|
| Underweight | 22.92% |
| Healthy weight | 26.50% |
| Overweight | 37.75% |
| Class 1 obesity | 46.50% |
| Class 2 obesity | 50.04% |
| Class 3 obesity | 55.07% |

### Smoking and food insecurity also matter

Observed diabetes rates:

- ever smoked 100 cigarettes: **14.03%**
- never smoked 100 cigarettes: **9.84%**
- food secure: **10.91%**
- low food security: **15.52%**
- very low food security: **17.28%**

This shows the target is not driven by one variable only. The disease outcome is associated with multiple lifestyle and socioeconomic factors.

## 5.4 Feature-set comparison

To check whether the signal really comes from lifestyle factors, I compared three feature groups.

### Diabetes (`DIBEV_A`)

| Feature set | ROC-AUC | PR-AUC |
|---|---:|---:|
| Demographics only | 0.732 | 0.219 |
| Lifestyle only | 0.721 | 0.246 |
| Combined | 0.783 | 0.296 |

### Hypertension (`HYPEV_A`)

| Feature set | ROC-AUC | PR-AUC |
|---|---:|---:|
| Demographics only | 0.772 | 0.619 |
| Lifestyle only | 0.683 | 0.552 |
| Combined | 0.799 | 0.669 |

### Interpretation

- For **diabetes**, lifestyle-only performance is close to demographics-only performance. This is exactly what we want for a lifestyle-focused assignment.
- For **hypertension**, demographics already explain a lot, especially age.
- For both targets, combining lifestyle and demographics performs best.

## 6. Why each next step is needed

The user asked not just for the steps, but for **why** each step is done and **how** it should be done. This section is written in that format.

## 6.1 Cleaning

### Why we need cleaning

Survey datasets are not ready for modeling as-is.

- Special values like `7`, `8`, `9`, `97`, `98`, and `99` often mean `Refused`, `Not Ascertained`, or `Don't Know`, not real values.
- Some variables are only asked of a subset of respondents, so naive missing-value handling can create false patterns.
- The readme explicitly warns that import tools can infer wrong types if too few rows are used.

### How we cleaned this dataset

We implemented the cleaning in `scripts/train_diabetes_model.py`.

1. We selected a compact diabetes-focused raw subset:
   - `AGEP_A`, `SEX_A`, `EDUCP_A`, `POVRATTC_A`, `BMICATD_A`, `SMKEV_A`, `ECIGEV_A`, `DRKSTAT_A`, `PA18_02R_A`, `SLPHOURS_A`, `FDSCAT3_A`, `DIBEV_A`, and `WTFA_A`
2. We converted survey non-response codes to actual missing values.
   - `SEX_A`: `7`, `9`
   - `EDUCP_A`: `97`, `99`
   - `BMICATD_A`: `9`
   - `SMKEV_A`, `ECIGEV_A`, `DIBEV_A`: `7`, `8`, `9`
   - `DRKSTAT_A`: `10`
   - `PA18_02R_A`: `8`
   - `SLPHOURS_A`: `97`, `98`, `99`
   - `FDSCAT3_A`: `8`
3. We dropped rows only when the target itself was invalid.
   - This reduced the modeling set from `32,629` to `32,600` rows.
4. We kept low-rate predictor missingness and handled it with imputation inside the modeling pipeline.
   - Why: dropping rows for 2% to 4% missingness would throw away too much data for little gain.
5. We kept `WTFA_A` in the engineered dataset.
   - Why: it is needed for population-representative prevalence estimates, even though the first machine-learning baseline was trained as an unweighted classifier.

### Why leakage removal matters

If the target is diabetes, then variables such as `DIBTYPE_A`, `DIBINS_A`, `DIBPILL_A`, `DIBINSTIME_A`, or diagnosis-age fields should not be used as predictors. They are consequences or follow-up questions after diagnosis, so they would make the model look unrealistically strong.

If the target is hypertension, then `HYP12M_A`, `HYPMED_A`, and `HYPDIF_A` should not be used as predictors for the same reason.

## 6.2 Feature engineering

### Why we need feature engineering

Machine-learning models need variables in a form they can learn from well. Raw survey codes are not ideal.

### How we engineered features

We moved from raw survey codes to a more interpretable feature set.

1. We created the binary target `target_diabetes` from `DIBEV_A`.
   - `1 -> diabetes`
   - `2 -> no diabetes`
2. We kept `age` as a numeric predictor and standardized it inside the model pipeline.
   - Why: age has a continuous relationship with chronic disease risk.
3. We converted high-cardinality or code-based variables into grouped categories:
   - `education_group` from `EDUCP_A`
     - `less_than_high_school` from codes `01-02`
     - `high_school_or_ged` from `03-04`
     - `some_college_or_associate` from `05-07`
     - `bachelors_or_higher` from `08-10`
   - `poverty_group` from `POVRATTC_A`
     - `below_poverty`, `near_poverty`, `middle_income`, `higher_income`
   - `bmi_group` from `BMICATD_A`
     - `under_or_healthy`, `overweight`, `class1_obesity`, `class2plus_obesity`
   - `activity_level` from `PA18_02R_A`
     - ordered activity categories based on the NHIS aerobic-guideline recode
   - `sleep_group` from `SLPHOURS_A`
     - `short_sleep` `<= 6`
     - `normal_sleep` `7-9`
     - `long_sleep` `>= 10`
4. We kept smoking, e-cigarette history, alcohol status, and food security as categorical behavioral features.
5. We used one-hot encoding for the categorical features.
   - Why: logistic regression and random forest both work better with explicit category indicators than with raw survey code numbers.

### Why I am not recommending heavy feature engineering immediately

The dataset is already rich. The first goal should be a strong, explainable baseline. Complex transformations should come after the baseline is stable.

That is exactly the order I followed:

1. build a compact, interpretable baseline first
2. verify that the baseline is non-leaky and stable
3. only then add a richer feature set and a stronger model to see whether there is still unused predictive signal

## 6.3 Model training

### Why we need more than one model

One model is not enough to justify a data science conclusion.

- A **logistic regression** baseline is interpretable and easy to explain.
- A **tree-based model** can capture non-linear interactions and category combinations that logistic regression may miss.
- Comparing them helps us decide whether complexity is actually necessary and whether the baseline is leaving obvious signal unused.

### How we trained

We implemented the training workflow in `scripts/train_diabetes_model.py`.

1. We used a **stratified 80/20 train/test split**.
   - Why: diabetes prevalence is only about `11.38%`, so stratification is needed to preserve class balance.
2. We ran **5-fold cross-validation on the training set**.
   - Why: one train/test split alone is too unstable for a reliable assignment conclusion.
3. We built a compact, interpretable baseline feature set first.
   - `age`, `sex`, grouped education, grouped poverty ratio, grouped BMI, smoking history, e-cigarette history, alcohol status, physical activity level, sleep group, and food security
   - Why: these variables are directly tied to the assignment topic and are easy to explain in a report.
4. We compared three baseline models on that compact feature set:
   - `DummyClassifier(strategy="prior")`
   - `LogisticRegression(class_weight="balanced")`
   - `RandomForestClassifier(class_weight="balanced_subsample")`
5. After the baseline was stable, we built a second, richer but still non-leaky feature set for score improvement.
   - Added `SMKNOW_A` to derive `smoking_status`
   - Added `PA18_05R_A` as a second physical-activity recode
   - Added `REGION`, `URBRRL23`, and `MARITAL_A`
   - Why: these variables add behavioral detail and contextual structure without using post-diagnosis diabetes variables.
6. We trained **CatBoost** on the richer feature set.
   - Why: CatBoost handles categorical features natively and often performs better on mixed survey-style tabular data than one-hot linear models.
7. We imputed missing values inside the scikit-learn baseline pipelines:
   - median for numeric features
   - most-frequent for categorical features
8. For CatBoost, we kept `age` numeric, filled missing age with the training median, and converted missing categorical values to an explicit `missing` category.
   - Why: CatBoost can use that structure directly without one-hot encoding.
9. We evaluated the models with ROC-AUC, PR-AUC, accuracy, and F1.

### Why these evaluation metrics are appropriate

- **ROC-AUC**: good for ranking ability across thresholds
- **PR-AUC**: especially important for imbalanced targets like diabetes
- **F1-score**: useful when positive cases matter and class imbalance exists
- **Accuracy alone is not enough**, because a model can look good by mostly predicting the majority class

### Why we improved the model after the baseline

Improving the model was necessary for three reasons:

1. to check whether the baseline was already close to the dataset ceiling or whether there was still recoverable signal
2. to test whether non-linear models and richer categorical structure help on this survey dataset
3. to make the final report stronger by separating the **best explanatory model** from the **best scoring model**

This is good practice in a data science assignment. If we never test a stronger alternative, we cannot tell whether the baseline result is genuinely solid or simply underfit.

### What the trained models actually did

| Model | Feature set | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | Compact interpretable | 0.500 | 0.114 | 0.500 | 0.114 | 0.886 | 0.000 |
| Logistic regression | Compact interpretable | 0.782 | 0.296 | 0.772 | 0.280 | 0.687 | 0.344 |
| Random forest | Compact interpretable | 0.781 | 0.290 | 0.768 | 0.274 | 0.732 | 0.356 |
| CatBoost | Richer non-leaky | 0.790 | 0.304 | 0.778 | 0.289 | 0.689 | 0.354 |

### What improved and why

The richer CatBoost model is the **best score model**.

- Compared with logistic regression, it improved test ROC-AUC from `0.772` to `0.778`.
- It improved test PR-AUC from `0.280` to `0.289`.
- It improved test F1 from `0.344` to `0.354`.

The improvement is not huge, but it is real and consistent with the cross-validation results. That tells us the baseline was already strong, and the richer model found a small amount of additional signal rather than completely changing the story.

### Which model should be presented as the final model?

There are two defensible answers, and they serve different purposes.

1. **Best explanatory model:** logistic regression
   - Why: easier to explain through coefficients, odds ratios, and permutation importance
   - Use this if the assignment prioritizes interpretability and clear reasoning
2. **Best score model:** richer CatBoost
   - Why: highest ROC-AUC and PR-AUC among the tested models
   - Use this if the assignment rewards predictive performance more heavily

My recommendation is to present both:

- use **logistic regression** as the main explanatory model in the report
- use **CatBoost** as evidence that you tested whether the score could be improved, and explain why the gain is modest rather than dramatic

### Why the dummy model matters

The dummy model has the **highest accuracy** at about `88.6%`, but that is misleading because it mostly predicts the majority class. Its ROC-AUC is `0.500`, PR-AUC is `0.114`, and F1 is `0.000`. This is the clearest demonstration that **accuracy alone would have led to the wrong conclusion**.

### Interpretation of the final logistic model

The strongest positive signals were:

- older age
- higher obesity category
- being below the poverty line
- being physically inactive
- very low food security
- lower educational attainment

Permutation importance on the hold-out set ranked the features in this order:

| Feature | Importance |
|---|---:|
| Age | 0.1416 |
| BMI group | 0.0529 |
| Poverty group | 0.0091 |
| Activity level | 0.0088 |
| Alcohol group | 0.0057 |
| Sex | 0.0045 |
| Food security | 0.0019 |
| Education group | 0.0016 |

This supports a clinically sensible story: **age and body composition are dominant, but socioeconomic and behavioral conditions add real predictive value**.

### Interpretation of the richer CatBoost model

CatBoost feature importance tells a similar story, but with slightly more nuance.

| Feature | Importance |
|---|---:|
| Age | 31.61 |
| BMI group | 14.36 |
| Alcohol group | 6.93 |
| Poverty group | 5.91 |
| Activity-guideline combo code | 5.54 |
| Region | 5.20 |
| Activity level | 4.92 |
| Sex | 4.23 |

This is a useful confirmation. The better-scoring model did **not** win by exploiting obviously inappropriate predictors. It still leans on the same main story:

- age and BMI dominate
- lifestyle behaviors matter
- socioeconomic context matters
- geographic and urban-rural context add a little extra signal

### Threshold behavior and why this is a screening model, not a diagnostic model

At the default `0.5` probability threshold, the logistic model produced:

- `TP = 534`
- `FP = 1833`
- `FN = 208`
- `TN = 3945`
- precision `0.226`
- recall `0.720`

This means the model catches many diabetes cases, but it also generates many false positives. That is acceptable for **risk screening** or prioritization, but not for medical diagnosis.

For the richer CatBoost model at the same `0.5` threshold:

- `TP = 556`
- `FP = 1840`
- `FN = 186`
- `TN = 3938`
- precision `0.232`
- recall `0.749`

This is slightly better screening behavior than the logistic baseline, but still far from diagnostic-grade performance. So the project conclusion stays the same: this is a **risk-screening model**, not a medical diagnosis model.

### Why we did not use survey weights in the model loss

We kept the NHIS survey weight for descriptive prevalence estimates, but the first predictive baseline was trained without survey weighting.

Why:

- the main assignment goal is row-level predictive discrimination
- scikit-learn baselines are easier to explain and reproduce this way
- weighted modeling can be added later as an extension, but it is not required for a defensible first project result

### Audit check: what was right and what needed fixing

What was already done correctly:

- the dataset choice was appropriate for the topic
- the target selection logic was defensible
- leakage-prone post-diagnosis diabetes variables were excluded
- class imbalance was handled with appropriate metrics instead of accuracy alone
- the baseline modeling workflow was valid and reproducible

What needed improvement:

- the original repo had **no `.gitignore`**, which is a problem before a GitHub push
- the first saved report documented the baseline well, but it did **not** yet document the model-improvement stage
- the stronger CatBoost benchmark had been tested separately, but it was not yet integrated into the saved training pipeline and artifacts

These issues are now fixed:

- `.gitignore` has been added
- the richer CatBoost benchmark is part of `scripts/train_diabetes_model.py`
- the saved outputs now include both baseline diagnostics and improved-model diagnostics

## 7. What I would recommend as the official assignment direction

### Recommended final framing

> **Primary project question:** Can lifestyle and basic demographic factors predict whether an adult has ever been diagnosed with diabetes in the 2024 NHIS Sample Adult dataset?

### Why this framing is strong

- It is specific.
- It matches the topic.
- The target exists directly in the data.
- The baseline analysis already shows the target is learnable.
- The result is interpretable for a report and presentation.
- The explainable logistic model achieved a **test ROC-AUC of 0.772** and **test PR-AUC of 0.280**.
- The richer CatBoost benchmark improved this to **0.778 ROC-AUC** and **0.289 PR-AUC**, so the assignment can show both explanation and attempted score improvement.

### Recommended secondary analysis

Use hypertension as a comparison target.

This gives the assignment an extra analytical layer:

- diabetes shows a stronger lifestyle-centered story
- hypertension shows a stronger overall predictive score but heavier age dependence

That comparison itself is a valuable insight.

## 8. If we fail to find strong predictive power later, why might that happen?

Even though the first-pass results are promising, there are important limitations.

### Possible reasons for limited final performance

- NHIS is **cross-sectional**, so it captures status, not disease onset over time.
- Many outcomes are **self-reported diagnoses**, not lab-confirmed disease.
- Access to healthcare affects diagnosis. Someone may have diabetes but not know it.
- Some important risk factors are missing or only weakly measured in self-report form.
- Survey recodes compress detail, which can reduce signal.
- If we choose the wrong features, leakage or over-cleaning can distort the result.

### Why that would still be a valid assignment result

A valid data science result is not only "the model worked." A valid result can also be:

> "The dataset is suitable for risk screening, but not for precise disease prediction because the available predictors are cross-sectional, self-reported, and partially post-diagnosis."

That is still a defensible conclusion if supported by analysis.

## 9. Saved artifacts

- Report: `docs/predictive-health-analysis.md`
- Git hygiene file: `.gitignore`
- Initial target-screening script: `scripts/nhis_target_screen.py`
- Full diabetes training pipeline: `scripts/train_diabetes_model.py`
- Engineered modeling dataset: `outputs/diabetes_model/engineered_diabetes_dataset.csv`
- Richer engineered modeling dataset: `outputs/diabetes_model/engineered_diabetes_dataset_richer.csv`
- Saved metrics: `outputs/diabetes_model/model_metrics.csv`
- Logistic coefficients: `outputs/diabetes_model/logistic_coefficients.csv`
- Logistic permutation importance: `outputs/diabetes_model/logistic_permutation_importance.csv`
- Threshold trade-off table: `outputs/diabetes_model/logistic_threshold_metrics.csv`
- CatBoost cross-validation folds: `outputs/diabetes_model/catboost_richer_cv_folds.csv`
- CatBoost feature importance: `outputs/diabetes_model/catboost_richer_feature_importance.csv`
- CatBoost threshold trade-off table: `outputs/diabetes_model/catboost_richer_threshold_metrics.csv`
- Saved trained models:
  - `outputs/diabetes_model/logistic_regression.joblib`
  - `outputs/diabetes_model/random_forest.joblib`
  - `outputs/diabetes_model/catboost_richer.cbm`

Run the scripts with:

```bash
source ~/python/bin/activate
python scripts/nhis_target_screen.py
python scripts/train_diabetes_model.py
```

## 10. Current conclusion

This dataset is suitable for the assignment.

- We **can** predict a chronic disease target from the available features.
- The strongest final choice is **diabetes**.
- The best **explanatory** model is **logistic regression**.
- The best **score** model is the richer **CatBoost** benchmark.
- The model is strong enough to justify a predictive-health assignment, but not strong enough to be described as a diagnostic tool.

## 11. Sources

- CDC NHIS 2024 documentation page: <https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html>
- CDC NHIS 2024 Sample Adult summary: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2024/Adult-summary.pdf>
- CDC NHIS 2024 Sample Adult codebook: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2024/adult-codebook.pdf>
- CDC BRFSS 2024 annual data page: <https://www.cdc.gov/brfss/annual_data/2024/llcp_multiq.html>
- CDC NHANES 2021-2023 data portal: <https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023>
