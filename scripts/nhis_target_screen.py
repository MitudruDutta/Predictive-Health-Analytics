from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.screening import compare_feature_sets, load_screening_frame, screen_candidate_targets, selected_feature_missingness


def main() -> None:
    df = load_screening_frame()
    missingness = (selected_feature_missingness(df) * 100).round(2)
    target_screening = screen_candidate_targets(df).round(4)
    feature_set_comparison = compare_feature_sets(df).round(4)

    print("Dataset shape:", df.shape)
    print("\nSelected feature missingness after recoding:")
    print(missingness.to_string())
    print("\nTarget screening:")
    print(target_screening.to_string(index=False))
    print("\nFeature-set comparison:")
    print(feature_set_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
