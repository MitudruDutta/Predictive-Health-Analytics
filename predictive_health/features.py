from __future__ import annotations

import numpy as np
import pandas as pd


def education_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value in [1, 2]:
        return "less_than_high_school"
    if value in [3, 4]:
        return "high_school_or_ged"
    if value in [5, 6, 7]:
        return "some_college_or_associate"
    if value in [8, 9, 10]:
        return "bachelors_or_higher"
    return np.nan


def poverty_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < 1:
        return "below_poverty"
    if value < 2:
        return "near_poverty"
    if value < 4:
        return "middle_income"
    return "higher_income"


def bmi_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value in [1, 2]:
        return "under_or_healthy"
    if value == 3:
        return "overweight"
    if value == 4:
        return "class1_obesity"
    if value in [5, 6]:
        return "class2plus_obesity"
    return np.nan


def alcohol_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    value = int(value)
    if value == 1:
        return "lifetime_abstainer"
    if value in [2, 3, 4]:
        return "former_drinker"
    if value in [5, 6, 7]:
        return "current_nonheavy"
    if value in [8, 9]:
        return "current_heavier_or_unknown"
    return np.nan


def sleep_group(value: float) -> str | float:
    if pd.isna(value):
        return np.nan
    if value <= 6:
        return "short_sleep"
    if value <= 9:
        return "normal_sleep"
    return "long_sleep"


def smoking_status(ever_smoked: float, smoking_now: float) -> str | float:
    if pd.isna(ever_smoked):
        return np.nan
    if ever_smoked == 2:
        return "never"
    if pd.isna(smoking_now):
        return "ever_unknown_current"
    if smoking_now in [1, 2]:
        return "current"
    if smoking_now == 3:
        return "former"
    return "ever_unknown_current"


def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(
        {
            "age": df["AGEP_A"],
            "sex": df["SEX_A"].map({1: "male", 2: "female"}),
            "education_group": df["EDUCP_A"].map(education_group),
            "poverty_group": df["POVRATTC_A"].map(poverty_group),
            "bmi_group": df["BMICATD_A"].map(bmi_group),
            "ever_smoked_100_cigs": df["SMKEV_A"].map({1: "yes", 2: "no"}),
            "ever_used_ecig": df["ECIGEV_A"].map({1: "yes", 2: "no"}),
            "alcohol_group": df["DRKSTAT_A"].map(alcohol_group),
            "activity_level": df["PA18_02R_A"].map(
                {1: "inactive", 2: "insufficiently_active", 3: "sufficiently_active"}
            ),
            "sleep_group": df["SLPHOURS_A"].map(sleep_group),
            "food_security": df["FDSCAT3_A"].map(
                {1: "food_secure", 2: "low_food_security", 3: "very_low_food_security"}
            ),
            "target_diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
            "survey_weight": df["WTFA_A"],
        }
    )
    return engineered[engineered["target_diabetes"].notna()].reset_index(drop=True)


def engineer_richer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.DataFrame(
        {
            "age": df["AGEP_A"],
            "sex": df["SEX_A"].map({1: "male", 2: "female"}),
            "education_group": df["EDUCP_A"].map(education_group),
            "poverty_group": df["POVRATTC_A"].map(poverty_group),
            "bmi_group": df["BMICATD_A"].map(bmi_group),
            "smoking_status": [
                smoking_status(ever_smoked, smoking_now)
                for ever_smoked, smoking_now in zip(df["SMKEV_A"], df["SMKNOW_A"])
            ],
            "ever_used_ecig": df["ECIGEV_A"].map({1: "yes", 2: "no"}),
            "alcohol_group": df["DRKSTAT_A"].map(alcohol_group),
            "activity_level": df["PA18_02R_A"].map(
                {1: "inactive", 2: "insufficiently_active", 3: "sufficiently_active"}
            ),
            "activity_guideline_combo_code": df["PA18_05R_A"].astype("Int64").astype(str),
            "sleep_group": df["SLPHOURS_A"].map(sleep_group),
            "food_security": df["FDSCAT3_A"].map(
                {1: "food_secure", 2: "low_food_security", 3: "very_low_food_security"}
            ),
            "region": df["REGION"].map({1: "northeast", 2: "midwest", 3: "south", 4: "west"}),
            "urban_rural": df["URBRRL23"].map(
                {
                    1: "large_central_metro",
                    2: "large_fringe_metro",
                    3: "medium_small_metro",
                    4: "nonmetro",
                }
            ),
            "marital_status": df["MARITAL_A"].map(
                {
                    1: "married_or_partnered",
                    2: "widowed_divorced_separated",
                    3: "never_married",
                }
            ),
            "target_diabetes": df["DIBEV_A"].map({1: 1, 2: 0}),
            "survey_weight": df["WTFA_A"],
        }
    )
    engineered["activity_guideline_combo_code"] = engineered[
        "activity_guideline_combo_code"
    ].replace("<NA>", np.nan)
    return engineered[engineered["target_diabetes"].notna()].reset_index(drop=True)

