from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import RICHER_OUTPUT_DIR
from predictive_health.etl import build_richer_diabetes_dataset, feature_columns
from predictive_health.io import ensure_dir, write_dataframe, write_text
from predictive_health.modeling import fit_and_evaluate_catboost, save_catboost_diagnostics


def main() -> None:
    ensure_dir(RICHER_OUTPUT_DIR)
    engineered_df = build_richer_diabetes_dataset()
    predictors = feature_columns(engineered_df)
    X = engineered_df[predictors]
    y = engineered_df["target_diabetes"].astype(int)

    metrics_row, cv_df, model, X_test, y_test = fit_and_evaluate_catboost(X, y)
    metrics_df = write_metrics(metrics_row)
    write_dataframe(metrics_df, RICHER_OUTPUT_DIR / "model_metrics.csv")
    save_catboost_diagnostics(RICHER_OUTPUT_DIR, model, X_test, y_test, cv_df)
    model.save_model(RICHER_OUTPUT_DIR / "catboost_richer.cbm")

    summary = "\n".join(
        [
            f"Engineered dataset rows: {len(engineered_df)}",
            f"Positive class rate: {y.mean():.4f}",
            "",
            "Richer CatBoost metrics:",
            metrics_df.to_string(index=False),
        ]
    )
    write_text(RICHER_OUTPUT_DIR / "summary.txt", summary)
    print(summary)


def write_metrics(metrics_row: dict[str, float | str]):
    import pandas as pd

    return pd.DataFrame([metrics_row]).round(4)


if __name__ == "__main__":
    main()

