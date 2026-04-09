from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictive_health.config import DATA_PATH, MISSING_CODE_MAP


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_raw_data(columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=columns)
    for column, values in MISSING_CODE_MAP.items():
        if column in df.columns:
            df[column] = df[column].replace(values, np.nan)
    return df


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_series(series: pd.Series, path: Path, index_label: str, value_name: str) -> None:
    ensure_dir(path.parent)
    series.rename(value_name).rename_axis(index_label).reset_index().to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
