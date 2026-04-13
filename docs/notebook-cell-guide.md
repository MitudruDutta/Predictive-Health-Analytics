# Notebook Cell-by-Cell Guide — Full Explanation

> Detailed companion for `predictive_health_pipeline.ipynb`.
> The notebook is **self-contained** — all helper functions, constants, and
> feature-engineering logic are defined inline (no external package imports).
> This guide covers what each cell does, how the code works, the mathematical
> basis behind every technique, and how to explain it to a professor.

---

## Cell 0 — Title & Objective (Markdown)

States the project goal:

- **Problem type:** Binary classification (diabetes = 1, no diabetes = 0).
- **Data source:** 2024 National Health Interview Survey (NHIS) — a nationally representative household survey conducted by the CDC. Each row is one adult respondent.
- **Feature philosophy:** We only use **lifestyle and demographic** variables (age, BMI, smoking, alcohol, exercise, etc.) — not lab results, genetics, or clinical measurements. This makes the model usable as a **screening tool** that requires no blood tests.

---

## Section 0 — Setup (Cells 1–16)

Since the notebook is standalone, all setup is done inline. The 614 lines of
setup code are split into **8 logical code cells** with markdown headers:

### Cell 2 (Code) — Imports

Standard Python data-science stack: `numpy`, `pandas`, `matplotlib`, `sklearn`,
`catboost`. All imports are collected in one place for clarity.

### Cell 3 (Code) — Constants

Defines the three key constants used throughout:

| Constant | Value | Purpose |
|---|---|---|
| `DATA_PATH` | `adult24.csv` | Path to the NHIS 2024 CSV (must be in same directory as notebook) |
| `RANDOM_STATE` | 42 | Seed for all random number generators. Ensures exact reproducibility. |
| `THRESHOLDS` | [0.5, 0.4, 0.3] | Decision boundaries for the threshold sweep. |

### Cell 4 (Code) — Column definitions & NHIS sentinel-code map

Defines which columns to read from the CSV, and the `MISSING_CODE_MAP` that
maps NHIS special codes (7=refused, 8=not ascertained, 9=don't know) to NaN.
Also defines three feature set groupings for the comparison in Section 2.

### Cell 6 (Code) — Data loading helpers

`load_raw_data(columns)` — reads selected CSV columns and replaces sentinel
codes with NaN. `load_screening_frame()` — convenience wrapper.

### Cell 8 (Code) — Target screening helpers

`screen_candidate_targets()`, `compare_feature_sets()`, etc. — functions used
in Section 2 to evaluate which disease target is most predictable.

### Cell 10 (Code) — Feature engineering functions

12 small transform functions that map raw NHIS codes to human-readable
categories (e.g., `bmi_group(3)` → `"overweight"`).

### Cell 12 (Code) — Dataset builders (ETL)

`build_baseline_diabetes_dataset()` and `build_richer_diabetes_dataset()` —
read the CSV, apply feature transforms, and return clean DataFrames.

### Cell 14 (Code) — Model-building helpers

`build_preprocessor()`, `evaluate_baseline_models()` — sklearn preprocessing
pipeline and the baseline model training/evaluation loop.

### Cell 16 (Code) — CatBoost helpers

`prepare_catboost_frame()`, `build_catboost_model()`, `fit_and_evaluate_catboost()`
— CatBoost-specific utilities for data prep, model creation, CV, and evaluation.

---

## Section 1 — Raw Data Load & Audit

### Cell 18 (Code) — Load screening frame

```python
screening_df = load_screening_frame()
```

**What `load_screening_frame()` does internally:**

1. Calls `pd.read_csv(DATA_PATH, usecols=columns)` — reads only the 17 columns we need from the 400+ column CSV. This is much faster and uses less memory than loading the full file.

2. Applies `MISSING_CODE_MAP` — NHIS uses special numeric codes for non-responses:
   - `7` = refused to answer
   - `8` = not ascertained (interviewer couldn't determine)
   - `9` = don't know
   - `97/98/99` = same meanings for multi-digit variables

   The code replaces these sentinel values with `NaN` (Python's "missing" marker):
   ```python
   df[column] = df[column].replace([7, 8, 9], np.nan)
   ```
   This is critical — without it, the model would treat "refused" (code 7) as a real category.

**Output:** 32,629 rows × 17 columns.

### Cell 19 (Code) — Missingness check

```python
selected_feature_missingness(screening_df) * 100
```

**How it works:** For each column, computes `column.isna().mean()` — the fraction of rows that are `NaN`. Multiply by 100 to get percentages.

**Why this matters:** If a feature is >20-30% missing, it may be unreliable. Here, the worst is 4.14% (food security) — all features are usable. Standard imputation (filling missing values) will handle the rest.

### Cell 21 (Code) — Sentinel age check (Section 1a)

```python
bad_age = age_raw['AGEP_A'].isin([97, 98, 99])
```

**The bug:** `MISSING_CODE_MAP` doesn't include `AGEP_A` (age), so sentinel codes 97/99 survive as real numbers. The model sees age=97 as "97 years old" instead of "refused to answer."

**Impact:** Only 52 out of 32,629 rows (0.16%) are affected. At this scale, the effect on model metrics is within noise. We flag it for intellectual honesty.

---

## Section 1b — Exploratory Data Analysis

### Cell 23 (Code) — Three EDA plots

**Plot 1 — Target class distribution:**

Shows the raw counts:
- ~28,890 adults **without** diabetes (class 0)
- ~3,710 adults **with** diabetes (class 1)
- **Imbalance ratio ≈ 7.8 : 1**

**Why imbalance matters mathematically:** A model that always predicts "no diabetes" achieves 88.6% accuracy — but catches zero diabetics. This is why we don't use accuracy as our primary metric. Instead we use:

- **ROC-AUC** (Area Under the Receiver Operating Characteristic curve): Measures how well the model ranks positives above negatives, regardless of threshold. Ranges from 0.5 (random) to 1.0 (perfect).
- **PR-AUC** (Area Under the Precision-Recall curve): Better than ROC-AUC for imbalanced data because it focuses on the minority class.

**Plot 2 — Age histogram by diabetes status:**

Two overlapping histograms. The diabetes group (orange) is shifted right — diabetics tend to be older. The separation is clear but incomplete: there are young diabetics and old non-diabetics.

**Mathematical insight:** If age were a *perfect* predictor, the two histograms would have zero overlap. The partial overlap tells us age is necessary but not sufficient — we need more features.

**Plot 3 — Diabetes rate by BMI group:**

A horizontal bar chart showing the proportion of diabetics in each BMI category:

| BMI Group | Diabetes Rate | Interpretation |
|---|---|---|
| Under/healthy weight | ~5% | Baseline risk |
| Overweight | ~10% | ~2× baseline |
| Class 1 obesity | ~16% | ~3× baseline |
| Class 2+ obesity | ~22% | ~4× baseline |

This is a clear **dose-response** relationship — the heavier the BMI category, the higher the diabetes rate. This validates BMI as a strong predictor.

---

## Section 2 — Target Screening

### Cell 25 (Code) — Screen 5 candidate targets

```python
target_screening = screen_candidate_targets(screening_df)
```

**What this function does step by step:**

1. **Builds 5 binary targets** from the raw survey variables:
   - Diabetes: `DIBEV_A` = 1 → positive, = 2 → negative
   - Prediabetes: `PREDIB_A` same coding
   - Hypertension: `HYPEV_A` same coding
   - Coronary heart disease: `CHDEV_A` same coding
   - Fair/poor health: `PHSTAT_A` ∈ {4,5} → positive, ∈ {1,2,3} → negative

2. **For each target:** fits a Logistic Regression with stratified 5-fold cross-validation and records ROC-AUC and PR-AUC.

**The math behind Logistic Regression (used for screening):**

Logistic Regression models the probability of the positive class as:

```
P(y=1 | x) = σ(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)
```

Where **σ** is the sigmoid function:

```
σ(z) = 1 / (1 + e^(-z))
```

This squashes any real number into the range (0, 1), giving us a probability. The weights **w** are learned by maximising the **log-likelihood** (equivalently, minimising the **log-loss / binary cross-entropy**):

```
L = -Σ [yᵢ·log(p̂ᵢ) + (1-yᵢ)·log(1-p̂ᵢ)]
```

**The math behind Stratified K-Fold Cross-Validation:**

The dataset is split into K=5 equal-sized folds. Each fold preserves the same class ratio as the full dataset (~11.4% positive). The model is trained on 4 folds and tested on the 1 remaining fold. This rotates 5 times, and the scores are averaged. This gives a reliable estimate of model performance without wasting data.

```
Score = (1/K) × Σ score_on_fold_k
```

**The math behind ROC-AUC:**

The ROC curve plots **True Positive Rate** (TPR = TP/(TP+FN), also called recall) against **False Positive Rate** (FPR = FP/(FP+TN)) at every possible threshold. The AUC is the area under this curve:

- AUC = 0.5 → model is no better than random
- AUC = 1.0 → model perfectly separates the classes
- AUC = 0.78 (our result) → the model ranks a randomly chosen diabetic above a randomly chosen non-diabetic 78% of the time

### Cell 26 (Code) — Feature set comparison

Tests three feature groupings on two targets:

| Set | Features | Idea |
|---|---|---|
| Demographics only | age, sex, education, poverty | Can demographics alone predict? |
| Lifestyle only | BMI, smoking, alcohol, exercise, sleep, food security | Can lifestyle alone predict? |
| Combined | All 11 | Does combining them help? |

**Result:** Combined always wins → both demographics AND lifestyle carry signal. Neither is sufficient alone.

---

## Section 3 — Feature Engineering (ETL)

### Cell 28 (Code) — Build engineered datasets

```python
baseline_df = build_baseline_diabetes_dataset()
richer_df   = build_richer_diabetes_dataset()
```

**What feature engineering does:**

Raw NHIS data uses numeric codes (e.g., `BMICATD_A = 3`). We transform these into human-readable categories:

```python
def bmi_group(value):
    if value in [1, 2]: return "under_or_healthy"
    if value == 3:      return "overweight"
    if value == 4:      return "class1_obesity"
    if value in [5, 6]: return "class2plus_obesity"
```

**Why we do this:**
1. **Interpretability** — "overweight" is easier to explain than "code 3"
2. **Grouping** — combines related codes (e.g., codes 5 and 6 both mean severe obesity)
3. **Non-linearity** — treating BMI as a category lets the model learn different effects for each group (e.g., obesity has a much larger effect than overweight)

**Baseline vs Richer feature sets:**

| Feature | Baseline | Richer | Transform |
|---|---|---|---|
| age | ✅ | ✅ | Raw number |
| sex | ✅ | ✅ | 1→male, 2→female |
| education_group | ✅ | ✅ | 4 groups |
| poverty_group | ✅ | ✅ | 4 income bins |
| bmi_group | ✅ | ✅ | 4 BMI categories |
| ever_smoked_100_cigs | ✅ | — | Binary |
| smoking_status | — | ✅ | 4 levels (never/current/former/unknown) |
| ever_used_ecig | ✅ | ✅ | Binary |
| alcohol_group | ✅ | ✅ | 4 groups |
| activity_level | ✅ | ✅ | 3 levels |
| activity_guideline_combo | — | ✅ | Categorical code |
| sleep_group | ✅ | ✅ | 3 groups (short/normal/long) |
| food_security | ✅ | ✅ | 3 levels |
| region | — | ✅ | 4 US Census regions |
| urban_rural | — | ✅ | 4 urbanization levels |
| marital_status | — | ✅ | 3 groups |

The richer set upgrades binary `ever_smoked` to a 4-level `smoking_status` by combining two survey variables — current smokers have different risk than former smokers.

### Cell 29 (Code) — Richer missingness

Same `isna().mean()` calculation as before, now on the engineered features. All under 5%.

### Section 3b (Markdown) — Survey weight note

`WTFA_A` is a **survey design weight**. Because the NHIS uses a complex multi-stage sampling design (not simple random sampling), each respondent represents a different number of people in the US population. The weight tells you how many.

**Why we don't use it in model training:** Our goal is **individual-level prediction** ("does this person have diabetes?"), not **population-level inference** ("what fraction of Americans have diabetes?"). Survey weights are for the latter. Instead, we handle class imbalance with `class_weight='balanced'`, which reweights the loss function by inverse class frequency.

### Cell 32 (Code) — Leakage check (Section 3a)

**What is data leakage?** When information from the target variable leaks into the features. For example, if "takes insulin" were a feature, it would perfectly predict diabetes — but that's circular, not a real prediction.

The check groups by each feature value and computes the diabetes rate. If any group has rate = 0% or 100%, it might indicate leakage. Only age=99 (sentinel code) and a tiny NaN group show 0% — not real leakage.

---

## Section 4 — Baseline Models

### Cell 34 (Code) — Train and evaluate

**Three models, each with a purpose:**

**1. DummyClassifier (`strategy='prior'`):**
Always predicts the majority class. Gets 88.6% accuracy (because 88.6% of people don't have diabetes). This is the **baseline to beat** — any real model must do better than this.

**2. Logistic Regression:**

Already explained in Section 2. Key settings:
- `max_iter=2000` — allows enough iterations for the optimiser to converge
- `class_weight='balanced'` — multiplies the loss for each class by `n_samples / (n_classes × n_samples_per_class)`:
  ```
  weight_positive = 32600 / (2 × 3710) ≈ 4.4
  weight_negative = 32600 / (2 × 28890) ≈ 0.56
  ```
  This makes each misclassified diabetic cost ~8× more than each misclassified non-diabetic, forcing the model to pay attention to the minority class.

**3. Random Forest:**

An **ensemble** of 500 decision trees. Each tree:
1. Gets a random subset of the data (bagging / bootstrap sampling)
2. At each split, considers a random subset of features
3. Splits on the feature and threshold that best separates the classes

The **final prediction** is the average of all 500 trees' predictions. This reduces overfitting because individual trees' errors cancel out.

Key settings:
- `n_estimators=500` — 500 trees
- `min_samples_leaf=10` — each leaf must have ≥10 samples (prevents memorising individual patients)
- `class_weight='balanced_subsample'` — like balanced, but recomputed for each tree's bootstrap sample

**The preprocessing pipeline (`ColumnTransformer`):**

Before any model is trained, features are preprocessed:

```
Numeric (age only):
  1. SimpleImputer(strategy='median')  → fills NaN with median age
  2. StandardScaler()                  → z-score: z = (x - mean) / std

Categorical (everything else):
  1. SimpleImputer(strategy='most_frequent')  → fills NaN with mode
  2. OneHotEncoder(handle_unknown='ignore')   → creates binary columns
```

**Why StandardScaler?** Logistic Regression uses gradient descent to optimise weights. If age (range 18-85) is on a different scale than one-hot features (0 or 1), the gradient descent takes zigzag paths and converges slowly. Scaling puts everything on the same scale.

**Why OneHotEncoder?** Categorical features like BMI group have no natural ordering (is "overweight" > "under_or_healthy"?). One-hot encoding creates a separate binary column for each category, so the model can learn independent weights for each.

### Cell 35 (Code) — Overfitting check

```python
gap = cv_roc_auc - test_roc_auc
```

**What overfitting is mathematically:** The model memorises the training data instead of learning general patterns. Symptom: training/CV score is much higher than test score.

A small positive gap (~0.01) is normal and healthy — it means the model generalises well. A gap >0.05 would be concerning. Our gaps are all under 0.015. ✅

### Cell 37 (Code) — Logistic coefficients (Section 4a)

**How to read logistic regression coefficients:**

The model computes a **log-odds** (also called logit):

```
log(p / (1-p)) = w₀ + w₁x₁ + w₂x₂ + ...
```

- **Coefficient (+0.91 for age):** Each 1-standard-deviation increase in age adds 0.91 to the log-odds of diabetes.
- **Odds ratio = e^coefficient:** e^0.91 = 2.49. This means each std-dev of age multiplies the odds of diabetes by 2.49×.
- **Negative coefficient (−0.83 for healthy BMI):** Being in the "under/healthy" BMI category reduces the odds by factor e^(−0.83) = 0.44 (a 56% reduction).

**Top risk factors found:**
1. Age (odds ratio 2.49) — the strongest single predictor
2. Class 2+ obesity (odds ratio 2.03)
3. Below poverty (odds ratio 1.50)

**Top protective factors:**
1. Healthy BMI (odds ratio 0.44)
2. Sufficiently active (odds ratio 0.77)
3. Higher income (odds ratio 0.73)

### Cell 38 (Code) — Permutation importance

**How it works:**

1. Compute the baseline ROC-AUC on the test set
2. For each feature: randomly shuffle that feature's values, recompute ROC-AUC
3. Importance = baseline AUC − shuffled AUC

If shuffling a feature causes a big drop in AUC, that feature is important. If shuffling has no effect, the feature is useless.

**Advantage over coefficients:** Permutation importance captures non-linear effects and interactions. Coefficients only measure linear relationships.

**Result:** Age (0.14 importance) and BMI (0.05) dominate. Smoking and e-cig use have near-zero importance — they don't help predictions much.

---

## Section 5 — CatBoost (Richer Features)

### Cell 40 (Code) — Train CatBoost

**What is CatBoost?**

CatBoost is a **gradient boosting** algorithm. The idea:

1. Start with a simple prediction (e.g., the log-odds of the positive class)
2. Compute the **residual errors** — how wrong is the current prediction for each sample?
3. Train a small decision tree to predict those residuals
4. Add that tree's predictions to the running total (scaled by the learning rate)
5. Repeat steps 2-4 for `iterations` rounds

**Mathematically:**

```
F₀(x) = log(p_positive / p_negative)           # initial prediction
Fₜ(x) = Fₜ₋₁(x) + η · hₜ(x)                   # add tree t's contribution
```

Where:
- `η = 0.03` is the **learning rate** (how much each tree contributes)
- `hₜ(x)` is the tree trained on the **negative gradient** of the loss function
- The loss function is **Logloss** (binary cross-entropy)

**Why CatBoost over regular gradient boosting (XGBoost, LightGBM)?**

CatBoost handles categorical features natively using **ordered target encoding**: for each sample, it computes the target mean from all *preceding* samples in a random permutation. This avoids the information leakage that standard target encoding causes.

**Key hyperparameters:**

| Parameter | Value | Effect |
|---|---|---|
| `iterations=800` | 800 boosting rounds | More rounds = more complex model |
| `depth=6` | Max tree depth | Shallow trees = less overfitting |
| `learning_rate=0.03` | Step size | Small = slow learning, less overfitting |
| `auto_class_weights='Balanced'` | Reweights loss | Same as Logistic Regression's balanced mode |
| `l2_leaf_reg=3` | L2 regularization on leaf values | Penalises large leaf predictions |

**`prepare_catboost_frame()` — why it's needed:**

CatBoost requires:
- No NaN in numeric columns → fill age NaN with median
- Categorical NaN explicitly marked → replace NaN with the string `"missing"` (treated as a new category)

### Cell 41 (Code) — Per-fold CV scores

Shows 5-fold CV scores for the CatBoost model. If one fold is much worse than others, it suggests instability. Here all 5 folds are within 0.78–0.80 → the model is stable.

### Cell 42 (Code) — CatBoost feature importance

CatBoost uses **PredictionValuesChange**: for each feature, sums the absolute changes in prediction values across all trees where that feature was used for splitting. Higher = more important.

**Difference from permutation importance:** This is computed from the tree structure itself (no re-evaluation needed), making it fast but specific to tree-based models.

### Cell 44 (Code) — Threshold sweep (Section 5a)

**The math behind thresholds:**

The model outputs a probability $\hat{p}$ for each sample. To make a binary decision, we compare $\hat{p}$ to a threshold $t$:

```
prediction = 1 if p̂ ≥ t else 0
```

**At threshold 0.5 (default):**
- Precision = TP/(TP+FP) = 556/2396 = 23.2% — of people *flagged* as diabetic, only 23% really are
- Recall = TP/(TP+FN) = 556/742 = 74.9% — of actual diabetics, we catch 75%

**At threshold 0.3:**
- Precision drops to 18.0% (more false alarms)
- Recall rises to 89.6% (catch nearly all diabetics)

**Trade-off:** In a clinical screening setting, missing a diabetic (false negative) is worse than a false alarm (false positive), so a lower threshold makes sense. The "right" threshold depends on the cost ratio of false negatives vs false positives.

---

## Section 6 — Hyperparameter Tuning

### Cell 46 (Code) — Define configs

Five CatBoost configurations exploring the regularisation-performance trade-off:

| Config | Key idea |
|---|---|
| current_baseline | Default settings from Section 5 |
| deeper_slower | Deeper trees (8), slower learning (0.02), more L2 (5) |
| balanced_medium | Middle ground |
| shallower_longer | Shallower trees (5), very slow learning (0.015), 1500 rounds |
| regularized_deep | Deep trees (8) with strong L2 (8) to prevent overfitting |

**The maths of L2 regularisation (`l2_leaf_reg`):**

L2 adds a penalty term to the loss function:

```
Total Loss = Logloss + λ × Σ(leaf_value²)
```

Where `λ = l2_leaf_reg`. Larger λ → leaf values are shrunk toward zero → model is simpler and less likely to overfit. The sweep varies λ from 3 to 8 to find the optimal trade-off.

### Cell 47 (Code) — CV helper function

```python
def cv_catboost(config, X, y):
```

**Uses 3-fold CV** (not 5) to save time during the sweep. For each fold:
1. Split data into train/validation
2. Prepare categorical features with `prepare_catboost_frame()`
3. Train CatBoost with the given config
4. Predict probabilities on validation set
5. Compute ROC-AUC and PR-AUC

Returns the **mean** across 3 folds.

**Why 3-fold is OK here:** The sweep is for *relative* ranking of configs, not absolute performance estimation. 3-fold is sufficient to rank them reliably while cutting compute time by 40%.

### Cell 48 (Code) — Run sweep

For each of the 5 configs:
1. Run `cv_catboost()` → get CV scores (used for selection)
2. Train on full training set → score on held-out test set (used for reporting)
3. Compute `cv_test_roc_gap` = CV score − test score (overfitting indicator)

**Critical methodological point:** The configs are sorted by `cv_roc_auc`, not `test_roc_auc`. Selecting by test score is a form of **data dredging** — you're effectively using the test set for model selection, which inflates the reported performance.

### Cell 50 (Code) — Pick winner (Section 6a)

The `shallower_longer` config wins:
- CV ROC-AUC: 0.7892
- Test ROC-AUC: 0.7793
- Gap: 0.0099 (very small → no overfitting)

**Key insight:** The best config only beats the baseline by 0.002 ROC-AUC. This means the **default CatBoost settings were already near-optimal** — there's no significant headroom left from hyperparameter tuning. The signal ceiling from lifestyle features alone is ~0.78.

---

## Section 7 — Final Comparison

### Cell 52 (Code) — Final scoreboard

| Model | Feature Set | Test ROC-AUC | Test PR-AUC | Notes |
|---|---|---|---|---|
| CatBoost | Richer (15 features) | 0.7777 | 0.2891 | Best score |
| Logistic Regression | Baseline (11 features) | 0.7723 | 0.2804 | Best interpretability |
| Random Forest | Baseline (11 features) | 0.7681 | 0.2736 | Slightly worse than LogReg |
| Dummy | Baseline | 0.5000 | 0.1138 | Random baseline |

### Cell 53 (Code) — Bar chart

Grouped bar chart showing CV vs. test ROC-AUC for all models. The dashed line at 0.5 marks random chance.

**How to read this chart:**
- All real models are well above 0.5 → they learned real patterns
- CV bars are slightly taller than test bars → small, healthy overfitting gap
- CatBoost's advantage over Logistic Regression is barely visible → the extra model complexity provides marginal benefit

---

## Key Takeaways for the Professor

1. **Why diabetes?** We didn't pick it arbitrarily — Section 2 systematically screened 5 targets and chose diabetes for its best balance of predictability and class imbalance.

2. **Why these features?** All features are non-invasive lifestyle/demographic variables available from a survey — no lab results needed. The combined set beats demographics-only and lifestyle-only (Section 2, Cell 13).

3. **Why multiple models?** Each serves a purpose:
   - Dummy → proves the data is learnable (beats random)
   - Logistic Regression → interpretable, coefficients have clinical meaning (odds ratios)
   - Random Forest → checks if non-linear patterns exist
   - CatBoost → pushes for maximum predictive performance

4. **Is it overfitting?** No. CV-to-test gaps are consistently ≤ 0.015. The model generalises well.

5. **What's the performance ceiling?** ~0.78 ROC-AUC. Lifestyle features alone cannot perfectly predict diabetes — genetics, lab results, and medical history would be needed to go higher. This is a known limitation and an intellectually honest result.

6. **Practical implication:** At a 0.3 threshold, the CatBoost model catches ~90% of diabetics. It could serve as a low-cost screening triage tool — flag high-risk individuals for a proper blood glucose test.

---

## Glossary of Key Terms

| Term | Meaning |
|---|---|
| **ROC-AUC** | Area Under the ROC Curve. Measures ranking quality. 0.5 = random, 1.0 = perfect. |
| **PR-AUC** | Area Under the Precision-Recall Curve. Better than ROC-AUC for imbalanced data. |
| **Precision** | Of all people predicted positive, what fraction truly are? TP/(TP+FP) |
| **Recall (Sensitivity)** | Of all truly positive people, what fraction did we catch? TP/(TP+FN) |
| **F1 Score** | Harmonic mean of precision and recall: 2×P×R/(P+R) |
| **Stratified Split** | Train/test split that preserves class proportions in both sets |
| **Cross-Validation (CV)** | Train on K-1 folds, test on the remaining fold, rotate K times |
| **Gradient Boosting** | Sequentially train weak learners, each correcting the previous ones' errors |
| **L2 Regularisation** | Penalty on large weights/leaf values: loss + λΣw² |
| **Odds Ratio** | e^(coefficient). How much one unit of a feature multiplies the odds |
| **Permutation Importance** | Drop in score when a feature is randomly shuffled |
| **Class Imbalance** | One class is much rarer than the other (here: 11.4% vs 88.6%) |
| **Log-loss (Cross-entropy)** | -Σ[y·log(p̂) + (1-y)·log(1-p̂)]. The loss function for binary classification |
| **Sigmoid** | σ(z) = 1/(1+e^(-z)). Converts log-odds to probability |
