from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictive_health.config import FINAL_OUTPUT_DIR
from predictive_health.etl import (
    build_baseline_diabetes_dataset,
    build_richer_diabetes_dataset,
    feature_columns,
)
from predictive_health.io import ensure_dir, write_dataframe, write_text
from predictive_health.modeling import (
    evaluate_baseline_models,
    fit_and_evaluate_catboost,
    save_baseline_models,
    save_catboost_diagnostics,
    save_logistic_diagnostics,
)


def main() -> None:
    ensure_dir(FINAL_OUTPUT_DIR)

    baseline_df = build_baseline_diabetes_dataset()
    write_dataframe(baseline_df, FINAL_OUTPUT_DIR / "engineered_diabetes_dataset.csv")
    baseline_predictors = feature_columns(baseline_df)
    X_baseline = baseline_df[baseline_predictors]
    y_baseline = baseline_df["target_diabetes"].astype(int)

    metrics_df, fitted_models, _, _, X_test_baseline, y_test_baseline = evaluate_baseline_models(
        X_baseline, y_baseline
    )
    save_logistic_diagnostics(FINAL_OUTPUT_DIR, fitted_models["logistic_regression"], X_test_baseline, y_test_baseline)
    save_baseline_models(FINAL_OUTPUT_DIR, fitted_models)

    richer_df = build_richer_diabetes_dataset()
    write_dataframe(richer_df, FINAL_OUTPUT_DIR / "engineered_diabetes_dataset_richer.csv")
    richer_predictors = feature_columns(richer_df)
    X_richer = richer_df[richer_predictors]
    y_richer = richer_df["target_diabetes"].astype(int)

    catboost_metrics, catboost_cv_df, catboost_model, X_test_richer, y_test_richer = fit_and_evaluate_catboost(
        X_richer, y_richer
    )
    save_catboost_diagnostics(FINAL_OUTPUT_DIR, catboost_model, X_test_richer, y_test_richer, catboost_cv_df)
    catboost_model.save_model(FINAL_OUTPUT_DIR / "catboost_richer.cbm")

    metrics_df = pd.concat([metrics_df, pd.DataFrame([catboost_metrics])], ignore_index=True)
    metrics_df = metrics_df.sort_values(by="test_roc_auc", ascending=False)
    write_dataframe(metrics_df, FINAL_OUTPUT_DIR / "model_metrics.csv")

    logistic_row = metrics_df.loc[metrics_df["model"] == "logistic_regression"].iloc[0]
    catboost_row = metrics_df.loc[metrics_df["model"] == "catboost_richer"].iloc[0]

    summary = "\n".join(
        [
            f"Baseline engineered dataset rows: {len(baseline_df)}",
            f"Baseline positive class rate: {y_baseline.mean():.4f}",
            f"Richer engineered dataset rows: {len(richer_df)}",
            "",
            "Model metrics:",
            metrics_df.round(4).to_string(index=False),
            "",
            (
                "Best score model: "
                f"CatBoost richer (test ROC-AUC {catboost_row['test_roc_auc']:.4f}, "
                f"test PR-AUC {catboost_row['test_pr_auc']:.4f})"
            ),
            (
                "Best interpretable model: "
                f"Logistic regression (test ROC-AUC {logistic_row['test_roc_auc']:.4f}, "
                f"test PR-AUC {logistic_row['test_pr_auc']:.4f})"
            ),
        ]
    )
    write_text(FINAL_OUTPUT_DIR / "summary.txt", summary)
    print(summary)
    print(f"\nSaved outputs to: {FINAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
