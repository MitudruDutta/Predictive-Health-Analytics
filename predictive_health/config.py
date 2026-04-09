from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "adult24.csv"
OUTPUT_ROOT = ROOT_DIR / "outputs"
FINAL_OUTPUT_DIR = OUTPUT_ROOT / "diabetes_model"
STAGE_OUTPUT_ROOT = OUTPUT_ROOT / "stages"
SCREENING_OUTPUT_DIR = STAGE_OUTPUT_ROOT / "01_screening"
ETL_OUTPUT_DIR = STAGE_OUTPUT_ROOT / "02_etl"
BASELINE_OUTPUT_DIR = STAGE_OUTPUT_ROOT / "03_baseline_models"
RICHER_OUTPUT_DIR = STAGE_OUTPUT_ROOT / "04_richer_model"

RANDOM_STATE = 42
THRESHOLDS = [0.5, 0.4, 0.3]

SCREENING_FEATURE_COLS = [
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

BASE_RAW_COLUMNS = SCREENING_FEATURE_COLS + ["DIBEV_A", "WTFA_A"]

RICHER_RAW_COLUMNS = [
    "AGEP_A",
    "SEX_A",
    "EDUCP_A",
    "POVRATTC_A",
    "BMICATD_A",
    "SMKEV_A",
    "SMKNOW_A",
    "ECIGEV_A",
    "DRKSTAT_A",
    "PA18_02R_A",
    "PA18_05R_A",
    "SLPHOURS_A",
    "FDSCAT3_A",
    "REGION",
    "URBRRL23",
    "MARITAL_A",
    "DIBEV_A",
    "WTFA_A",
]

MISSING_CODE_MAP = {
    "SEX_A": [7, 9],
    "EDUCP_A": [97, 99],
    "BMICATD_A": [9],
    "SMKEV_A": [7, 8, 9],
    "SMKNOW_A": [7, 8, 9],
    "ECIGEV_A": [7, 8, 9],
    "DRKSTAT_A": [10],
    "PA18_02R_A": [8],
    "PA18_05R_A": [8],
    "SLPHOURS_A": [97, 98, 99],
    "FDSCAT3_A": [8],
    "MARITAL_A": [7, 8, 9],
    "DIBEV_A": [7, 8, 9],
    "PREDIB_A": [7, 8, 9],
    "HYPEV_A": [7, 8, 9],
    "CHDEV_A": [7, 8, 9],
    "PHSTAT_A": [7, 8, 9],
}

FEATURE_SET_MAP = {
    "demographics_only": ["AGEP_A", "SEX_A", "EDUCP_A", "POVRATTC_A"],
    "lifestyle_only": [
        "BMICATD_A",
        "SMKEV_A",
        "ECIGEV_A",
        "DRKSTAT_A",
        "PA18_02R_A",
        "SLPHOURS_A",
        "FDSCAT3_A",
    ],
    "combined": SCREENING_FEATURE_COLS,
}

