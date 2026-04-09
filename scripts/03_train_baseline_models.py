from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import BASELINE_OUTPUT_DIR
from predictive_health.etl import build_baseline_diabetes_dataset, feature_columns
from predictive_health.io import ensure_dir, write_dataframe, write_text
from predictive_health.modeling import (
    evaluate_baseline_models,
    save_baseline_models,
    save_logistic_diagnostics,
)


def main() -> None:
    ensure_dir(BASELINE_OUTPUT_DIR)
    engineered_df = build_baseline_diabetes_dataset()
    predictors = feature_columns(engineered_df)
    X = engineered_df[predictors]
    y = engineered_df["target_diabetes"].astype(int)

    metrics_df, fitted_models, _, _, X_test, y_test = evaluate_baseline_models(X, y)
    write_dataframe(metrics_df.round(4), BASELINE_OUTPUT_DIR / "model_metrics.csv")
    save_logistic_diagnostics(BASELINE_OUTPUT_DIR, fitted_models["logistic_regression"], X_test, y_test)
    save_baseline_models(BASELINE_OUTPUT_DIR, fitted_models)

    summary = "\n".join(
        [
            f"Engineered dataset rows: {len(engineered_df)}",
            f"Positive class rate: {y.mean():.4f}",
            "",
            "Baseline model metrics:",
            metrics_df.round(4).to_string(index=False),
        ]
    )
    write_text(BASELINE_OUTPUT_DIR / "summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()

