from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import ETL_OUTPUT_DIR
from predictive_health.etl import (
    build_baseline_diabetes_dataset,
    build_richer_diabetes_dataset,
    predictor_missingness,
)
from predictive_health.io import ensure_dir, write_dataframe, write_series, write_text


def main() -> None:
    ensure_dir(ETL_OUTPUT_DIR)
    baseline_df = build_baseline_diabetes_dataset()
    richer_df = build_richer_diabetes_dataset()
    baseline_missingness = (predictor_missingness(baseline_df) * 100).round(2)
    richer_missingness = (predictor_missingness(richer_df) * 100).round(2)

    write_dataframe(baseline_df, ETL_OUTPUT_DIR / "baseline_diabetes_dataset.csv")
    write_dataframe(richer_df, ETL_OUTPUT_DIR / "richer_diabetes_dataset.csv")
    write_series(
        baseline_missingness,
        ETL_OUTPUT_DIR / "baseline_predictor_missingness.csv",
        index_label="variable",
        value_name="missing_percent",
    )
    write_series(
        richer_missingness,
        ETL_OUTPUT_DIR / "richer_predictor_missingness.csv",
        index_label="variable",
        value_name="missing_percent",
    )

    summary = "\n".join(
        [
            f"Baseline engineered dataset rows: {len(baseline_df)}",
            f"Baseline positive class rate: {baseline_df['target_diabetes'].mean():.4f}",
            "",
            "Baseline predictor missingness (%):",
            baseline_missingness.to_string(),
            "",
            f"Richer engineered dataset rows: {len(richer_df)}",
            f"Richer positive class rate: {richer_df['target_diabetes'].mean():.4f}",
            "",
            "Richer predictor missingness (%):",
            richer_missingness.to_string(),
        ]
    )
    write_text(ETL_OUTPUT_DIR / "summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()

