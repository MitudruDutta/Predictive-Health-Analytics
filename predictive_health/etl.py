from __future__ import annotations

import pandas as pd

from predictive_health.config import BASE_RAW_COLUMNS, RICHER_RAW_COLUMNS
from predictive_health.features import engineer_baseline_features, engineer_richer_features
from predictive_health.io import load_raw_data


def build_baseline_diabetes_dataset() -> pd.DataFrame:
    raw_df = load_raw_data(BASE_RAW_COLUMNS)
    return engineer_baseline_features(raw_df)


def build_richer_diabetes_dataset() -> pd.DataFrame:
    raw_df = load_raw_data(RICHER_RAW_COLUMNS)
    return engineer_richer_features(raw_df)


def predictor_missingness(engineered_df: pd.DataFrame) -> pd.Series:
    predictor_columns = [
        column for column in engineered_df.columns if column not in {"target_diabetes", "survey_weight"}
    ]
    return engineered_df[predictor_columns].isna().mean().sort_values(ascending=False)


def feature_columns(engineered_df: pd.DataFrame) -> list[str]:
    return [column for column in engineered_df.columns if column not in {"target_diabetes", "survey_weight"}]

