# Step-By-Step Understanding Guide

This document is written to help you properly understand the full project, not just run it.

At every stage, the same four questions are answered:

1. What are we doing?
2. Why are we doing it?
3. How are we doing it?
4. What comes out of that stage?

---

## 1. Big picture

### What are we trying to do?

We are trying to answer this question:

> Can lifestyle and basic demographic factors predict whether an adult has ever been diagnosed with diabetes in the 2024 NHIS Sample Adult dataset?

This is a classification problem.

- Input: lifestyle and demographic variables
- Output: diabetes status, yes or no

### Why is this a good assignment question?

Because it is:

- meaningful in public health
- easy to explain
- supported by real data
- suitable for a full data science pipeline

This is important. A good project is not just about getting a score. It is about asking a question that the data can realistically answer.

---

## 2. The full workflow in one view

| Stage | Main question | Main script | Main output |
|---|---|---|---|
| Problem framing | Is this topic worth doing? | documented in `docs/` | clear project question |
| Target screening | Which disease target should we predict? | `scripts/01_target_screening.py` | target comparison tables |
| ETL and cleaning | How do we make the raw survey usable? | `scripts/02_etl_diabetes_datasets.py` | engineered datasets |
| Baseline modeling | Can a simple model predict diabetes at all? | `scripts/03_train_baseline_models.py` | baseline metrics |
| Improvement stage | Can we improve score without cheating or leaking? | `scripts/04_train_richer_catboost.py` | improved metrics |
| Final consolidated run | Can we save the full final result in one place? | `scripts/train_diabetes_model.py` | final artifacts |

The workflow is ordered this way on purpose.

We do **not** begin with the most complex model. We begin with problem clarity.

---

## 3. Step 1: Problem framing

### What are we doing?

We are choosing the problem carefully before building any model.

### Why are we doing it?

Because a vague project creates weak analysis.

If we start with:

> "Let’s predict something health-related"

that is too broad.

Instead, we need:

- one clear target
- one clear predictor set
- one clear reason the problem matters

### How are we doing it?

We frame the project around:

- topic: lifestyle and chronic disease
- data source: NHIS 2024 Sample Adult
- target: diagnosed diabetes
- prediction type: binary classification

### What comes out of this stage?

A precise research question and a defensible project direction.

That is why the project is not just "health analytics." It is specifically diabetes risk classification from lifestyle and demographic features.

---

## 4. Step 2: Understanding the dataset

### What are we doing?

We are checking whether the dataset is suitable before modeling.

### Why are we doing it?

Because not every dataset that looks health-related is actually usable for this topic.

We need to confirm:

- does the dataset contain chronic-disease outcomes?
- does it contain lifestyle predictors?
- is the sample size large enough?
- is missingness manageable?

### How are we doing it?

We inspect the NHIS 2024 Sample Adult dataset:

- rows: `32,629`
- columns: `630`
- file: `data/adult24.csv`

It contains:

- disease targets such as `DIBEV_A`, `HYPEV_A`, `PREDIB_A`, `CHDEV_A`
- lifestyle variables such as BMI, smoking, alcohol, physical activity, sleep, and food security

### What comes out of this stage?

A decision that the dataset is strong enough for the assignment.

This step matters because it prevents using a weak or unsuitable dataset just because it was available.

---

## 5. Step 3: Target screening

### What are we doing?

We are checking which disease target is the best one to model.

### Why are we doing it?

Because the dataset has multiple possible outcomes, and we should not choose one blindly.

A good target should be:

- relevant to the topic
- present in enough cases to model properly
- predictable from lifestyle-related features
- easy to explain in the report

### How are we doing it?

We use `scripts/01_target_screening.py`.

That script:

1. loads a compact set of relevant features
2. recodes survey missing values
3. tests several target variables
4. uses a quick logistic-regression screening model
5. compares ROC-AUC and PR-AUC across targets

Targets screened:

- diabetes
- prediabetes
- hypertension
- coronary heart disease
- fair or poor self-rated health

### Why logistic regression for screening?

Because screening is not the final modeling stage.

At this point we only need a fast, reasonable, interpretable method to compare targets under a common setup.

### What did we learn?

- Hypertension scored very strongly overall.
- Diabetes also scored well.
- Diabetes fits the assignment topic better because the lifestyle story is cleaner.

### What comes out of this stage?

The decision to make **diabetes** the main target and **hypertension** the comparison target.

That is one of the most important decisions in the whole project.

---

## 6. Step 4: ETL and cleaning

### What are we doing?

We are converting the raw NHIS survey file into modeling datasets.

### Why are we doing it?

Because the raw file is not directly usable.

Problems in raw survey data:

- coded missing values such as `7`, `8`, `9`, `97`, `98`, `99`
- too many columns for the assignment
- variable codes that are hard to interpret directly

### What does ETL mean here?

- **Extract** the columns we need
- **Transform** them into valid modeling variables
- **Load** the cleaned and engineered datasets for later stages

### How are we doing it?

We use:

- `predictive_health/io.py`
- `predictive_health/features.py`
- `predictive_health/etl.py`
- `scripts/02_etl_diabetes_datasets.py`

This stage builds two datasets:

1. **Baseline compact dataset**
   - designed for explanation and interpretability
2. **Richer dataset**
   - designed for score improvement

### Why do we recode missing survey codes to real missing values?

Because codes like `9` or `99` are not actual category values. They mean things such as:

- refused
- don’t know
- not ascertained

If we leave them as normal values, the model learns false patterns.

### Why do we not drop lots of rows?

Because most predictor missingness is low.

If we delete too many rows:

- we waste useful data
- we make the minority class even smaller
- we reduce model stability

So we only remove rows when the **target itself** is invalid.

### Why is this a good decision?

Because the target must be known for supervised learning, but predictor missingness can often be handled later with imputation.

### What comes out of this stage?

Two engineered diabetes datasets and missingness summaries.

Outputs are saved in:

- `outputs/stages/02_etl/`

---

## 7. Step 5: Feature engineering

### What are we doing?

We are turning raw survey variables into more meaningful predictors.

### Why are we doing it?

Because raw codes are hard to interpret and not always ideal for modeling.

Feature engineering helps us:

- simplify categories
- make variables easier to explain
- reduce noise
- align the predictors with the assignment topic

### How are we doing it?

Examples:

- `EDUCP_A` becomes `education_group`
- `POVRATTC_A` becomes `poverty_group`
- `BMICATD_A` becomes `bmi_group`
- `SLPHOURS_A` becomes `sleep_group`
- `PA18_02R_A` becomes `activity_level`

In the richer dataset we also add:

- `smoking_status`
- `activity_guideline_combo_code`
- `region`
- `urban_rural`
- `marital_status`

### Why build a compact feature set first?

Because we want a model that we can explain clearly.

If we jump directly to a more complex feature set, it becomes harder to answer:

- what the model learned
- why the model learned it
- whether the gain is real or just complexity

### Why build a richer feature set later?

Because once the baseline is stable, we want to test whether more structure improves the score.

That is proper modeling discipline:

- first explainable
- then stronger

### What comes out of this stage?

Two engineered feature tables:

- one for explanation
- one for improvement

---

## 8. Step 6: Train/test design

### What are we doing?

We are setting up a fair evaluation method before comparing models.

### Why are we doing it?

Because model scores are meaningless if the evaluation setup is weak.

### How are we doing it?

We use:

- stratified 80/20 train/test split
- 5-fold cross-validation on the training set

### Why stratified splitting?

Because diabetes is an imbalanced target.

Only about `11.38%` of rows are positive.

If we split without stratification, one side may accidentally get too many or too few positives.

### Why cross-validation?

Because a single split can be lucky or unlucky.

Cross-validation gives a more stable estimate of how the model behaves across multiple training and validation slices.

### What comes out of this stage?

A fair comparison setup for the models.

---

## 9. Step 7: Choosing evaluation metrics

### What are we doing?

We are choosing the metrics that will define model quality.

### Why are we doing it?

Because accuracy alone would be misleading here.

If the model predicts mostly "no diabetes," accuracy can still look high because the negative class is much larger.

### How are we doing it?

We use:

- **ROC-AUC**
- **PR-AUC**
- **F1-score**
- **Accuracy**

### Why each metric matters

#### ROC-AUC

Measures ranking quality across thresholds.

Use it to answer:

> Does the model generally rank higher-risk people above lower-risk people?

#### PR-AUC

More useful when the positive class is rare.

Use it to answer:

> How well does the model focus on real positives instead of being overwhelmed by negatives?

#### F1-score

Balances precision and recall.

Use it to answer:

> Is the model finding positives without collapsing into too many wrong alarms or too many misses?

#### Accuracy

Still reported, but not trusted by itself.

Use it only as supporting context.

### What comes out of this stage?

A correct evaluation philosophy.

This is why the dummy model matters so much: it proves that accuracy can lie.

---

## 10. Step 8: Baseline model training

### What are we doing?

We are training simple reference models first.

### Why are we doing it?

Because before we try to maximize score, we need to prove that:

- the target is learnable
- the pipeline works
- a simple interpretable model already captures meaningful signal

### How are we doing it?

We use `scripts/03_train_baseline_models.py`.

Models trained:

- `DummyClassifier`
- `LogisticRegression`
- `RandomForestClassifier`

### Why these three?

#### Dummy classifier

Purpose:

- no-skill baseline

Why:

- shows how a trivial model behaves
- proves whether our real models are actually useful

#### Logistic regression

Purpose:

- interpretable baseline

Why:

- coefficients can be explained
- odds ratios can be discussed
- good default for tabular classification

#### Random forest

Purpose:

- moderate-complexity non-linear baseline

Why:

- checks whether non-linear relationships matter
- gives a stronger comparison without going straight to gradient boosting

### What did we learn?

The compact logistic model achieved:

- ROC-AUC `0.7723`
- PR-AUC `0.2804`

That is strong enough to support the project.

The dummy model had high accuracy but failed on meaningful classification quality. That proves why accuracy alone is not acceptable.

### What comes out of this stage?

Baseline metrics and interpretable diagnostics.

Outputs are saved in:

- `outputs/stages/03_baseline_models/`

---

## 11. Step 9: Interpreting the baseline model

### What are we doing?

We are trying to understand what the baseline model learned.

### Why are we doing it?

Because a data science assignment is not only about prediction. It is also about explanation.

### How are we doing it?

We save:

- logistic coefficients
- odds-ratio-style interpretation
- permutation importance
- threshold tables

### What did the baseline model emphasize?

The strongest signals were:

- age
- BMI group
- poverty group
- activity level
- alcohol group

### Why is this useful?

Because it tells a coherent health story.

The model is not behaving randomly. It is using variables that make sense clinically and socially.

### What comes out of this stage?

A model that can be defended in a report, presentation, or viva.

---

## 12. Step 10: Model improvement

### What are we doing?

We are testing whether a stronger model and a richer feature set can improve the score.

### Why are we doing it?

Because if we stop at the baseline, we do not know whether:

- the baseline is already close to the best achievable performance
- or we left useful signal unused

This stage is about checking that.

### How are we doing it?

We use `scripts/04_train_richer_catboost.py`.

Changes made:

- use the richer feature set
- use CatBoost instead of only linear or bagged-tree models

### Why CatBoost?

Because it is strong on categorical tabular data.

That matches this project well, since many predictors are grouped categories.

### What did we learn?

The richer CatBoost model achieved:

- ROC-AUC `0.7777`
- PR-AUC `0.2891`

This is better than the logistic baseline, but not by a huge margin.

### Why is a small improvement still valuable?

Because it tells us something important:

- the baseline was already good
- the richer model found some extra signal
- the dataset does have more structure, but not enough to completely transform performance

### What comes out of this stage?

A better-scoring model and a stronger final project story.

Outputs are saved in:

- `outputs/stages/04_richer_model/`

### Can we improve the score even more?

Only a little, and even that depends on which metric you care about most.

We ran an additional CatBoost tuning sweep after the saved richer model.

What happened:

- a tuned `shallower_longer` setup improved ROC-AUC slightly to `0.7793`
- but the saved current richer CatBoost still kept the best PR-AUC at `0.2891`

This is a very important conclusion.

It means the model is **not dramatically under-tuned**. The remaining improvement from basic tuning is small and mostly changes the trade-off between metrics instead of changing the whole project outcome.

So if someone asks, "Can the score still improve?", the honest answer is:

> Yes, slightly. But not enough to change the main conclusion of the project.

---

## 13. Step 11: Final model choice

### What are we doing?

We are deciding how to present the final result.

### Why are we doing it?

Because "best model" can mean two different things:

- best at explanation
- best at score

### How are we handling that?

We keep both answers:

#### Best explanatory model

- logistic regression

Why:

- easier to explain
- easier to justify
- easier to interpret in a report

#### Best score model

- richer CatBoost

Why:

- best ROC-AUC
- best PR-AUC

### What should you say in the assignment?

Say that:

- logistic regression is the main explanatory model
- CatBoost is the best performance model
- the small gap between them shows the baseline was already strong

That is a much better answer than pretending one model solves everything.

---

## 14. Step 12: Why this is a screening model, not a diagnostic model

### What are we doing?

We are setting the correct interpretation boundary.

### Why are we doing it?

Because it would be scientifically wrong to claim that this model diagnoses diabetes.

### How do we know that?

- the data is cross-sectional
- the diagnoses are self-reported
- the model still has many false positives
- the predictor set is not a medical diagnostic panel

### What should the model be called instead?

A **risk-screening model** or **risk-prioritization model**.

That is the correct language.

---

## 15. Step 13: What files you should look at to understand each stage

### For problem framing

- `docs/01-problem-statement-and-topic-choice.md`

### For target selection

- `scripts/01_target_screening.py`
- `outputs/stages/01_screening/summary.txt`

### For ETL

- `scripts/02_etl_diabetes_datasets.py`
- `predictive_health/etl.py`
- `outputs/stages/02_etl/summary.txt`

### For feature engineering

- `predictive_health/features.py`
- `docs/04-feature-engineering.md`

### For baseline training

- `scripts/03_train_baseline_models.py`
- `outputs/stages/03_baseline_models/model_metrics.csv`

### For improved modeling

- `scripts/04_train_richer_catboost.py`
- `outputs/stages/04_richer_model/model_metrics.csv`

### For final combined outputs

- `scripts/train_diabetes_model.py`
- `outputs/diabetes_model/summary.txt`

---

## 16. Step 14: How to explain the entire project in one paragraph

If you need a short explanation for a report or viva, use this structure:

> We first checked whether the NHIS 2024 Sample Adult dataset was suitable for a lifestyle-and-chronic-disease project. Then we screened multiple disease targets and selected diabetes as the primary outcome because it matched the topic well and showed meaningful predictive signal. After that, we performed ETL by extracting relevant raw columns, converting survey non-response codes into missing values, and building engineered features such as BMI group, poverty group, activity level, and sleep group. We trained baseline models including a dummy classifier, logistic regression, and random forest to establish a fair reference point. Finally, we tested a richer CatBoost model to see whether performance could improve without introducing leakage. The logistic model remained the best explanatory model, while the richer CatBoost model achieved the best predictive score.

---

## 17. Final understanding

If you remember only one thing, remember this:

This project is not just "train a model on health data."

It is a structured workflow:

1. define the problem
2. verify that the dataset supports it
3. choose the right target
4. clean the data correctly
5. engineer meaningful features
6. evaluate simple models first
7. improve only after the baseline is understood
8. interpret the result honestly

That is what makes the project defensible.
