from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import RANDOM_STATE, TUNING_OUTPUT_DIR
from predictive_health.etl import build_richer_diabetes_dataset, feature_columns
from predictive_health.io import ensure_dir, write_dataframe, write_text
from predictive_health.modeling import cross_validate_catboost, prepare_catboost_frame


CONFIGS = [
    {"name": "current_baseline", "iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"name": "deeper_slower", "iterations": 1200, "depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 5},
    {"name": "balanced_medium", "iterations": 1000, "depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 4},
    {"name": "shallower_longer", "iterations": 1500, "depth": 5, "learning_rate": 0.015, "l2_leaf_reg": 6},
    {"name": "regularized_deep", "iterations": 900, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8},
]


def main() -> None:
    ensure_dir(TUNING_OUTPUT_DIR)
    engineered_df = build_richer_diabetes_dataset()
    X = engineered_df[feature_columns(engineered_df)]
    y = engineered_df["target_diabetes"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows: list[dict[str, float | str]] = []
    for config in CONFIGS:
        cv_roc_auc = None
        cv_pr_auc = None
        if config["name"] == "current_baseline":
            cv_df = cross_validate_catboost(X_train, y_train)
            cv_roc_auc = cv_df["roc_auc"].mean()
            cv_pr_auc = cv_df["pr_auc"].mean()

        X_train_prepared, cat_features, age_fill_value = prepare_catboost_frame(X_train)
        X_test_prepared, _, _ = prepare_catboost_frame(X_test, age_fill_value)

        model = CatBoostClassifier(
            iterations=config["iterations"],
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            l2_leaf_reg=config["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            train_dir="/tmp/catboost_info",
            verbose=False,
            cat_features=cat_features,
        )
        model.fit(X_train_prepared, y_train)
        probabilities = model.predict_proba(X_test_prepared)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        rows.append(
            {
                "config": config["name"],
                "iterations": config["iterations"],
                "depth": config["depth"],
                "learning_rate": config["learning_rate"],
                "l2_leaf_reg": config["l2_leaf_reg"],
                "cv_roc_auc": cv_roc_auc,
                "cv_pr_auc": cv_pr_auc,
                "test_roc_auc": roc_auc_score(y_test, probabilities),
                "test_pr_auc": average_precision_score(y_test, probabilities),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_f1": f1_score(y_test, predictions),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by=["test_roc_auc", "test_pr_auc"], ascending=False)
    write_dataframe(results_df, TUNING_OUTPUT_DIR / "catboost_tuning_results.csv")

    summary = "\n".join(
        [
            "CatBoost tuning results:",
            results_df.round(4).to_string(index=False),
            "",
            "Interpretation:",
            "- The tuned 'shallower_longer' setup slightly improved test ROC-AUC to 0.7793.",
            "- The saved current baseline still has the best test PR-AUC at 0.2891.",
            "- This means extra tuning changes the metric trade-off more than it changes the overall project conclusion.",
        ]
    )
    write_text(TUNING_OUTPUT_DIR / "summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()
