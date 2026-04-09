from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import SCREENING_OUTPUT_DIR
from predictive_health.io import ensure_dir, write_dataframe, write_series, write_text
from predictive_health.screening import compare_feature_sets, load_screening_frame, screen_candidate_targets, selected_feature_missingness


def main() -> None:
    ensure_dir(SCREENING_OUTPUT_DIR)
    df = load_screening_frame()
    missingness = (selected_feature_missingness(df) * 100).round(2)
    target_screening = screen_candidate_targets(df).round(4)
    feature_set_comparison = compare_feature_sets(df).round(4)

    write_series(
        missingness,
        SCREENING_OUTPUT_DIR / "selected_feature_missingness.csv",
        index_label="variable",
        value_name="missing_percent",
    )
    write_dataframe(target_screening, SCREENING_OUTPUT_DIR / "target_screening.csv")
    write_dataframe(feature_set_comparison, SCREENING_OUTPUT_DIR / "feature_set_comparison.csv")

    summary = "\n".join(
        [
            f"Dataset shape: {df.shape}",
            "",
            "Selected feature missingness (%):",
            missingness.to_string(),
            "",
            "Target screening:",
            target_screening.to_string(index=False),
            "",
            "Feature-set comparison:",
            feature_set_comparison.to_string(index=False),
        ]
    )
    write_text(SCREENING_OUTPUT_DIR / "summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()

