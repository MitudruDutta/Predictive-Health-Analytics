from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/adult24.csv")
OUTPUT_DIR = Path("outputs/diabetes_model")
RANDOM_STATE = 42
THRESHOLDS = [0.5, 0.4, 0.3]

BASE_RAW_COLUMNS = [
    "AGEP_A",
    "SEX_A",
    "EDUCP_A",
    "POVRATTC_A",
    "BMICATD_A",
    "SMKEV_A",
    "ECIGEV_A",
    "DRKSTAT_A",
    "PA18_02R_A",
    "SLPHOURS_A",
    "FDSCAT3_A",
    "DIBEV_A",
    "WTFA_A",
]

RICHER_RAW_COLUMNS = [
    "AGEP_A",
    "SEX_A",
    "EDUCP_A",
    "POVRATTC_A",
    "BMICATD_A",
    "SMKEV_A",
    "SMKNOW_A",
    "ECIGEV_A",
    "DRKSTAT_A",
    "PA18_02R_A",
    "PA18_05R_A",
    "SLPHOURS_A",
    "FDSCAT3_A",
    "REGION",
    "URBRRL23",
    "MARITAL_A",
    "DIBEV_A",
    "WTFA_A",
]

MISSING_CODE_MAP = {
    "SEX_A": [7, 9],
    "EDUCP_A": [97, 99],
    "BMICATD_A": [9],
    "SMKEV_A": [7, 8, 9],
    "SMKNOW_A": [7, 8, 9],
    "ECIGEV_A": [7, 8, 9],
    "DRKSTAT_A": [10],
    "PA18_02R_A": [8],
    "PA18_05R_A": [8],
    "SLPHOURS_A": [97, 98, 99],
    "FDSCAT3_A": [8],
    "MARITAL_A": [7, 8, 9],
    "DIBEV_A": [7, 8, 9],
}


def load_raw_data(columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=columns)
    for column, values in MISSING_CODE_MAP.items():
        if column in df.columns:
            df[column] = df[column].replace(values, np.nan)
    return df


def education_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value in [1, 2]:
        return "less_than_high_school"
    if value in [3, 4]:
        return "high_school_or_ged"
    if value in [5, 6, 7]:
        return "some_college_or_associate"
    if value in [8, 9, 10]:
        return "bachelors_or_higher"
    return np.nan


def poverty_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < 1:
        return "below_poverty"
    if value < 2:
        return "near_poverty"
    if value < 4:
        return "middle_income"
    return "higher_income"


def bmi_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value in [1, 2]:
        return "under_or_healthy"
    if value == 3:
        return "overweight"
    if value == 4:
        return "class1_obesity"
    if value in [5, 6]:
        return "class2plus_obesity"
    return np.nan


def alcohol_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value == 1:
        return "lifetime_abstainer"
    if value in [2, 3, 4]:
        return "former_drinker"
    if value in [5, 6, 7]:
        return "current_nonheavy"
    if value in [8, 9]:
        return "current_heavier_or_unknown"
    return np.nan


def sleep_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value <= 6:
        return "short_sleep"
    if value <= 9:
        return "normal_sleep"
    return "long_sleep"


def smoking_status(ever_smoked: float, smoking_now: float) -> str | float:
    if pd.isna(ever_smoked):
        return np.nan
    if ever_smoked == 2:
        return "never"
    if pd.isna(smoking_now):
        return "ever_unknown_current"
    if smoking_now in [1, 2]:
        return "current"
    if smoking_now == 3:
        return "former"
    return "ever_unknown_current"


def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(
        {
            "age": df["AGEP_A"],
            "sex": df["SEX_A"].map({1: "male", 2: "female"}),
            "education_group": df["EDUCP_A"].map(education_group),
            "poverty_group": df["POVRATTC_A"].map(poverty_group),
            "bmi_group": df["BMICATD_A"].map(bmi_group),
            "ever_smoked_100_cigs": df["SMKEV_A"].map({1: "yes", 2: "no"}),
            "ever_used_ecig": df["ECIGEV_A"].map({1: "yes", 2: "no"}),
            "alcohol_group": df["DRKSTAT_A"].map(alcohol_group),
            "activity_level": df["PA18_02R_A"].map(
                {1: "inactive", 2: "insufficiently_active", 3: "sufficiently_active"}
            ),
            "sleep_group": df["SLPHOURS_A"].map(sleep_group),
            "food_security": df["FDSCAT3_A"].map(
                {1: "food_secure", 2: "low_food_security", 3: "very_low_food_security"}
            ),
            "target_diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
            "survey_weight": df["WTFA_A"],
        }
    )
    return engineered[engineered["target_diabetes"].notna()].reset_index(drop=True)


def engineer_richer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(
        {
            "age": df["AGEP_A"],
            "sex": df["SEX_A"].map({1: "male", 2: "female"}),
            "education_group": df["EDUCP_A"].map(education_group),
            "poverty_group": df["POVRATTC_A"].map(poverty_group),
            "bmi_group": df["BMICATD_A"].map(bmi_group),
            "smoking_status": [
                smoking_status(ever_smoked, smoking_now)
                for ever_smoked, smoking_now in zip(df["SMKEV_A"], df["SMKNOW_A"])
            ],
            "ever_used_ecig": df["ECIGEV_A"].map({1: "yes", 2: "no"}),
            "alcohol_group": df["DRKSTAT_A"].map(alcohol_group),
            "activity_level": df["PA18_02R_A"].map(
                {1: "inactive", 2: "insufficiently_active", 3: "sufficiently_active"}
            ),
            "activity_guideline_combo_code": df["PA18_05R_A"].astype("Int64").astype(str),
            "sleep_group": df["SLPHOURS_A"].map(sleep_group),
            "food_security": df["FDSCAT3_A"].map(
                {1: "food_secure", 2: "low_food_security", 3: "very_low_food_security"}
            ),
            "region": df["REGION"].map({1: "northeast", 2: "midwest", 3: "south", 4: "west"}),
            "urban_rural": df["URBRRL23"].map(
                {
                    1: "large_central_metro",
                    2: "large_fringe_metro",
                    3: "medium_small_metro",
                    4: "nonmetro",
                }
            ),
            "marital_status": df["MARITAL_A"].map(
                {
                    1: "married_or_partnered",
                    2: "widowed_divorced_separated",
                    3: "never_married",
                }
            ),
            "target_diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
            "survey_weight": df["WTFA_A"],
        }
    )
    engineered["activity_guideline_combo_code"] = engineered[
        "activity_guideline_combo_code"
    ].replace("<NA>", np.nan)
    return engineered[engineered["target_diabetes"].notna()].reset_index(drop=True)


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [column for column in feature_columns if column == "age"]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_model_pipelines(feature_columns: list[str]) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(feature_columns)
    return {
        "dummy": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_sklearn_models(
    feature_frame: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, pipeline in build_model_pipelines(feature_frame.columns.tolist()).items():
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "average_precision", "accuracy", "f1"],
        )

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)

        metrics_rows.append(
            {
                "model": model_name,
                "feature_set": "compact_interpretable",
                "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
                "cv_pr_auc": cv_scores["test_average_precision"].mean(),
                "cv_accuracy": cv_scores["test_accuracy"].mean(),
                "cv_f1": cv_scores["test_f1"].mean(),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
                "test_pr_auc": average_precision_score(y_test, probabilities),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_f1": f1_score(y_test, predictions),
            }
        )
        fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="test_roc_auc", ascending=False)
    return metrics_df, fitted_models, X_train, y_train, X_test, y_test


def prepare_catboost_frame(
    feature_frame: pd.DataFrame, age_fill_value: float | None = None
) -> tuple[pd.DataFrame, list[str], float]:
    prepared = feature_frame.copy()
    if age_fill_value is None:
        age_fill_value = prepared["age"].median()
    prepared["age"] = prepared["age"].fillna(age_fill_value)
    categorical_columns = [column for column in prepared.columns if column != "age"]
    for column in categorical_columns:
        prepared[column] = prepared[column].astype(object).where(prepared[column].notna(), "missing")
    return prepared, categorical_columns, float(age_fill_value)


def build_catboost_model(cat_features: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        train_dir=str(Path("/tmp/catboost_info")),
        verbose=False,
        cat_features=cat_features,
    )


def cross_validate_catboost(feature_frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_index, valid_index) in enumerate(cv.split(feature_frame, target), start=1):
        X_fold_train = feature_frame.iloc[train_index]
        X_fold_valid = feature_frame.iloc[valid_index]
        y_fold_train = target.iloc[train_index]
        y_fold_valid = target.iloc[valid_index]

        X_fold_train_prepared, cat_features, age_fill_value = prepare_catboost_frame(X_fold_train)
        X_fold_valid_prepared, _, _ = prepare_catboost_frame(X_fold_valid, age_fill_value)

        model = build_catboost_model(cat_features)
        model.fit(X_fold_train_prepared, y_fold_train)

        probabilities = model.predict_proba(X_fold_valid_prepared)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "roc_auc": roc_auc_score(y_fold_valid, probabilities),
                "pr_auc": average_precision_score(y_fold_valid, probabilities),
                "accuracy": accuracy_score(y_fold_valid, predictions),
                "f1": f1_score(y_fold_valid, predictions),
            }
        )

    return pd.DataFrame(fold_rows)


def fit_and_evaluate_catboost(
    feature_frame: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float | str], pd.DataFrame, CatBoostClassifier, pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv_df = cross_validate_catboost(X_train, y_train)

    X_train_prepared, cat_features, age_fill_value = prepare_catboost_frame(X_train)
    X_test_prepared, _, _ = prepare_catboost_frame(X_test, age_fill_value)

    model = build_catboost_model(cat_features)
    model.fit(X_train_prepared, y_train)

    probabilities = model.predict_proba(X_test_prepared)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics_row = {
        "model": "catboost_richer",
        "feature_set": "richer_non_leaky",
        "cv_roc_auc": cv_df["roc_auc"].mean(),
        "cv_pr_auc": cv_df["pr_auc"].mean(),
        "cv_accuracy": cv_df["accuracy"].mean(),
        "cv_f1": cv_df["f1"].mean(),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
        "test_pr_auc": average_precision_score(y_test, probabilities),
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
    }
    return metrics_row, cv_df, model, X_test_prepared, y_test


def save_logistic_diagnostics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": model.named_steps["model"].coef_[0],
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
    coefficients.sort_values(by="abs_coefficient", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_coefficients.csv",
        index=False,
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    permutation_df = pd.DataFrame(
        {
            "feature_name": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    permutation_df.to_csv(OUTPUT_DIR / "logistic_permutation_importance.csv", index=False)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        threshold_rows.append(
            {
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "logistic_threshold_metrics.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
        }
    ).to_csv(OUTPUT_DIR / "logistic_test_predictions.csv", index=False)


def save_catboost_diagnostics(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_df: pd.DataFrame,
) -> None:
    importance_df = model.get_feature_importance(prettified=True).rename(
        columns={"Feature Id": "feature_name", "Importances": "importance"}
    )
    importance_df.to_csv(OUTPUT_DIR / "catboost_richer_feature_importance.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "catboost_richer_cv_folds.csv", index=False)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        threshold_rows.append(
            {
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "catboost_richer_threshold_metrics.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
        }
    ).to_csv(OUTPUT_DIR / "catboost_richer_test_predictions.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_raw_df = load_raw_data(BASE_RAW_COLUMNS)
    baseline_engineered_df = engineer_baseline_features(baseline_raw_df)
    baseline_engineered_df.to_csv(OUTPUT_DIR / "engineered_diabetes_dataset.csv", index=False)

    baseline_feature_columns = [
        column for column in baseline_engineered_df.columns if column not in {"target_diabetes", "survey_weight"}
    ]
    X_baseline = baseline_engineered_df[baseline_feature_columns]
    y_baseline = baseline_engineered_df["target_diabetes"].astype(int)

    metrics_df, fitted_models, _, _, X_test_baseline, y_test_baseline = evaluate_sklearn_models(
        X_baseline, y_baseline
    )
    save_logistic_diagnostics(fitted_models["logistic_regression"], X_test_baseline, y_test_baseline)

    for model_name in ["logistic_regression", "random_forest"]:
        joblib.dump(fitted_models[model_name], OUTPUT_DIR / f"{model_name}.joblib")

    richer_raw_df = load_raw_data(RICHER_RAW_COLUMNS)
    richer_engineered_df = engineer_richer_features(richer_raw_df)
    richer_engineered_df.to_csv(OUTPUT_DIR / "engineered_diabetes_dataset_richer.csv", index=False)

    richer_feature_columns = [
        column for column in richer_engineered_df.columns if column not in {"target_diabetes", "survey_weight"}
    ]
    X_richer = richer_engineered_df[richer_feature_columns]
    y_richer = richer_engineered_df["target_diabetes"].astype(int)

    catboost_metrics, catboost_cv_df, catboost_model, X_test_richer, y_test_richer = fit_and_evaluate_catboost(
        X_richer, y_richer
    )
    save_catboost_diagnostics(catboost_model, X_test_richer, y_test_richer, catboost_cv_df)
    catboost_model.save_model(OUTPUT_DIR / "catboost_richer.cbm")

    metrics_df = pd.concat([metrics_df, pd.DataFrame([catboost_metrics])], ignore_index=True)
    metrics_df = metrics_df.sort_values(by="test_roc_auc", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    logistic_row = metrics_df.loc[metrics_df["model"] == "logistic_regression"].iloc[0]
    catboost_row = metrics_df.loc[metrics_df["model"] == "catboost_richer"].iloc[0]
    summary_lines = [
        f"Baseline engineered dataset rows: {len(baseline_engineered_df)}",
        f"Baseline positive class rate: {y_baseline.mean():.4f}",
        f"Richer engineered dataset rows: {len(richer_engineered_df)}",
        "",
        "Model metrics:",
        metrics_df.round(4).to_string(index=False),
        "",
        (
            "Best score model: "
            f"CatBoost richer (test ROC-AUC {catboost_row['test_roc_auc']:.4f}, "
            f"test PR-AUC {catboost_row['test_pr_auc']:.4f})"
        ),
        (
            "Best interpretable model: "
            f"Logistic regression (test ROC-AUC {logistic_row['test_roc_auc']:.4f}, "
            f"test PR-AUC {logistic_row['test_pr_auc']:.4f})"
        ),
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
