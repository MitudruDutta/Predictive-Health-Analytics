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

from predictive_health.config import RANDOM_STATE, THRESHOLDS
from predictive_health.io import ensure_dir


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


def build_baseline_model_pipelines(feature_columns: list[str]) -> dict[str, Pipeline]:
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


def evaluate_baseline_models(
    feature_frame: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, pipeline in build_baseline_model_pipelines(feature_frame.columns.tolist()).items():
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
        rows.append(
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
    metrics_df = pd.DataFrame(rows).sort_values(by="test_roc_auc", ascending=False)
    return metrics_df, fitted_models, X_train, y_train, X_test, y_test


def save_logistic_diagnostics(output_dir: Path, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    ensure_dir(output_dir)
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
        output_dir / "logistic_coefficients.csv",
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
    permutation_df.to_csv(output_dir / "logistic_permutation_importance.csv", index=False)

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
    pd.DataFrame(threshold_rows).to_csv(output_dir / "logistic_threshold_metrics.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
        }
    ).to_csv(output_dir / "logistic_test_predictions.csv", index=False)


def prepare_catboost_frame(
    feature_frame: pd.DataFrame,
    age_fill_value: float | None = None,
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
        X_train = feature_frame.iloc[train_index]
        X_valid = feature_frame.iloc[valid_index]
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]
        X_train_prepared, cat_features, age_fill_value = prepare_catboost_frame(X_train)
        X_valid_prepared, _, _ = prepare_catboost_frame(X_valid, age_fill_value)
        model = build_catboost_model(cat_features)
        model.fit(X_train_prepared, y_train)
        probabilities = model.predict_proba(X_valid_prepared)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "pr_auc": average_precision_score(y_valid, probabilities),
                "accuracy": accuracy_score(y_valid, predictions),
                "f1": f1_score(y_valid, predictions),
            }
        )
    return pd.DataFrame(fold_rows)


def fit_and_evaluate_catboost(
    feature_frame: pd.DataFrame,
    target: pd.Series,
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


def save_catboost_diagnostics(
    output_dir: Path,
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_df: pd.DataFrame,
) -> None:
    ensure_dir(output_dir)
    importance_df = model.get_feature_importance(prettified=True).rename(
        columns={"Feature Id": "feature_name", "Importances": "importance"}
    )
    importance_df.to_csv(output_dir / "catboost_richer_feature_importance.csv", index=False)
    cv_df.to_csv(output_dir / "catboost_richer_cv_folds.csv", index=False)

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
    pd.DataFrame(threshold_rows).to_csv(output_dir / "catboost_richer_threshold_metrics.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
        }
    ).to_csv(output_dir / "catboost_richer_test_predictions.csv", index=False)


def save_baseline_models(output_dir: Path, fitted_models: dict[str, Pipeline]) -> None:
    ensure_dir(output_dir)
    for model_name in ["logistic_regression", "random_forest"]:
        joblib.dump(fitted_models[model_name], output_dir / f"{model_name}.joblib")
