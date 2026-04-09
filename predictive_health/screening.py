from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_health.config import FEATURE_SET_MAP, RANDOM_STATE, SCREENING_FEATURE_COLS, TARGET_COLS
from predictive_health.io import load_raw_data


def load_screening_frame() -> pd.DataFrame:
    return load_raw_data(SCREENING_FEATURE_COLS + TARGET_COLS)


def build_screening_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_cols = [col for col in feature_cols if col in {"AGEP_A", "POVRATTC_A", "SLPHOURS_A"}]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def weighted_binary_prevalence(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna()
    y = series.loc[mask].astype(float)
    w = weights.loc[mask]
    return float((y * w).sum() / w.sum())


def build_target_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
        "prediabetes": df["PREDIB_A"].map({1: 1, 2: 0}),
        "hypertension": df["HYPEV_A"].map({1: 1, 2: 0}),
        "coronary_heart_disease": df["CHDEV_A"].map({1: 1, 2: 0}),
        "fair_or_poor_health": df["PHSTAT_A"].map(
            lambda x: 1 if x in {4, 5} else (0 if x in {1, 2, 3} else np.nan)
        ),
    }


def selected_feature_missingness(df: pd.DataFrame) -> pd.Series:
    return df[SCREENING_FEATURE_COLS].isna().mean().sort_values(ascending=False)


def screen_candidate_targets(df: pd.DataFrame) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_screening_pipeline(SCREENING_FEATURE_COLS)
    rows: list[dict[str, float | int | str]] = []
    for name, target in build_target_map(df).items():
        mask = target.notna()
        X = df.loc[mask, SCREENING_FEATURE_COLS]
        y = target.loc[mask].astype(int)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
        rows.append(
            {
                "target": name,
                "n": len(y),
                "positive_rate": y.mean(),
                "weighted_positive_rate": weighted_binary_prevalence(y, df.loc[mask, "WTFA_A"]),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "pr_auc_mean": scores["test_average_precision"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(by="roc_auc_mean", ascending=False)


def compare_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    target_map = build_target_map(df)
    for target_name in ["hypertension", "diabetes"]:
        target = target_map[target_name]
        mask = target.notna()
        y = target.loc[mask].astype(int)
        for feature_set_name, feature_cols in FEATURE_SET_MAP.items():
            X = df.loc[mask, feature_cols]
            pipeline = build_screening_pipeline(feature_cols)
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
            rows.append(
                {
                    "target": target_name,
                    "feature_set": feature_set_name,
                    "roc_auc_mean": scores["test_roc_auc"].mean(),
                    "pr_auc_mean": scores["test_average_precision"].mean(),
                }
            )
    return pd.DataFrame(rows)

