from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/adult24.csv"

FEATURE_COLS = [
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
]

TARGET_COLS = ["DIBEV_A", "PREDIB_A", "HYPEV_A", "CHDEV_A", "PHSTAT_A", "WTFA_A"]

SPECIAL_CODES = {
    "SEX_A": [7, 9],
    "EDUCP_A": [97, 99],
    "BMICATD_A": [9],
    "SMKEV_A": [7, 8, 9],
    "ECIGEV_A": [7, 8, 9],
    "DRKSTAT_A": [10],
    "PA18_02R_A": [8],
    "SLPHOURS_A": [97, 98, 99],
    "FDSCAT3_A": [8],
    "DIBEV_A": [7, 8, 9],
    "PREDIB_A": [7, 8, 9],
    "HYPEV_A": [7, 8, 9],
    "CHDEV_A": [7, 8, 9],
    "PHSTAT_A": [7, 8, 9],
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=FEATURE_COLS + TARGET_COLS)
    for column, codes in SPECIAL_CODES.items():
        df[column] = df[column].replace(codes, np.nan)
    return df


def build_pipeline(feature_cols: list[str]) -> Pipeline:
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


def main() -> None:
    df = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    target_map = {
        "diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
        "prediabetes": df["PREDIB_A"].map({1: 1, 2: 0}),
        "hypertension": df["HYPEV_A"].map({1: 1, 2: 0}),
        "coronary_heart_disease": df["CHDEV_A"].map({1: 1, 2: 0}),
        "fair_or_poor_health": df["PHSTAT_A"].map(lambda x: 1 if x in {4, 5} else (0 if x in {1, 2, 3} else np.nan)),
    }

    print("Dataset shape:", df.shape)
    print("\nSelected feature missingness after recoding:")
    print((df[FEATURE_COLS].isna().mean() * 100).sort_values(ascending=False).round(2).to_string())

    print("\nTarget screening")
    print("target,n,positive_rate,weighted_positive_rate,roc_auc_mean,pr_auc_mean")
    combined_pipeline = build_pipeline(FEATURE_COLS)
    for name, target in target_map.items():
        mask = target.notna()
        X = df.loc[mask, FEATURE_COLS]
        y = target.loc[mask].astype(int)
        scores = cross_validate(combined_pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
        weighted_prev = weighted_binary_prevalence(y, df.loc[mask, "WTFA_A"])
        print(
            f"{name},{len(y)},{y.mean():.4f},{weighted_prev:.4f},"
            f"{scores['test_roc_auc'].mean():.4f},{scores['test_average_precision'].mean():.4f}"
        )

    feature_sets = {
        "demographics_only": ["AGEP_A", "SEX_A", "EDUCP_A", "POVRATTC_A"],
        "lifestyle_only": ["BMICATD_A", "SMKEV_A", "ECIGEV_A", "DRKSTAT_A", "PA18_02R_A", "SLPHOURS_A", "FDSCAT3_A"],
        "combined": FEATURE_COLS,
    }

    print("\nFeature-set comparison")
    print("target,feature_set,roc_auc_mean,pr_auc_mean")
    for target_name in ["hypertension", "diabetes"]:
        target = target_map[target_name]
        mask = target.notna()
        y = target.loc[mask].astype(int)
        for feature_set_name, feature_cols in feature_sets.items():
            X = df.loc[mask, feature_cols]
            pipeline = build_pipeline(feature_cols)
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
            print(
                f"{target_name},{feature_set_name},"
                f"{scores['test_roc_auc'].mean():.4f},{scores['test_average_precision'].mean():.4f}"
            )


if __name__ == "__main__":
    main()
