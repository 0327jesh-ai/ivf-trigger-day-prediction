# --------------------------------------------------
# Feature Engineering for IVF Trigger Prediction
# --------------------------------------------------
# Purpose:
# - Transform raw clinical features into meaningful ML features
# - Apply SAME logic in training, API, and batch inference
# --------------------------------------------------

import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw IVF trigger dataset.

    Parameters:
        df (pd.DataFrame): Raw clinical data

    Returns:
        pd.DataFrame: Feature-engineered dataset
    """
    # --------------------------------------------------
    # CREATE MISSING VALUE FLAGS
    # --------------------------------------------------
    # Purpose: Capture information from missing data itself
    # Usefulness: ML model may learn that missingness is predictive (hidden clinical pattern)
    for col in ["AMH (ng/mL)", "Estradiol_pg_mL", "Progesterone_ng_mL"]:
        if col in df.columns:
            # Create binary column: 1 if value was missing, 0 if present
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # --------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------
    # Purpose: Ensure ML pipeline does not fail due to missing numeric data
    # Usefulness: Maintains data integrity, avoids NaN errors in calculations, preserves patterns for ML

    numeric_cols = [
        "Age", 
        "AMH (ng/mL)", 
        "Day", 
        "Follicle_Count", 
        "Estradiol_pg_mL", 
        "Progesterone_ng_mL"
    ]  
    # Columns to check and fill missing
    for col in numeric_cols:
        if col in df.columns:
            # Replace missing numeric values with median
            # Purpose: Fill missing with typical value to prevent errors
            # Usefulness: Ensures feature engineering (e.g., ratios, bins) works
            df[col] = df[col].fillna(df[col].median())
    
    # --------------------------------------------------
    #  Drop non-informative features
    # --------------------------------------------------
    # Patient_ID is only for identification, not prediction
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    # --------------------------------------------------
    # AGE → Age Group + Age Risk Flag
    # --------------------------------------------------
    # Medical logic:
    # Fertility response changes non-linearly with age

    # Age grouping (captures fertility phases)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 35, 40, 100],
        labels=["<30", "30-35", "35-40", "40+"]
    )

    # Binary risk indicator for advanced age
    df["Age_Risk"] = (df["Age"] >= 35).astype(int)

    # --------------------------------------------------
    # AMH → AMH Level + Low AMH Flag
    # --------------------------------------------------
    # Medical logic:
    # AMH reflects ovarian reserve, best interpreted in ranges

    df["AMH_Level"] = pd.cut(
        df["AMH (ng/mL)"],
        bins=[0, 1, 3, 100],
        labels=["Low", "Normal", "High"]
    )

    # Flag for poor ovarian reserve
    df["Low_AMH"] = (df["AMH (ng/mL)"] < 1).astype(int)

    # --------------------------------------------------
    # STIMULATION DAY → Day Zone
    # --------------------------------------------------
    # Medical logic:
    # Trigger usually happens around day 9–12

    df["Day_Zone"] = pd.cut(
        df["Day"],
        bins=[0, 8, 10, 12, 20],
        labels=["Early", "Mid", "Ideal", "Late"]
    )

    # --------------------------------------------------
    # FOLLICLE COUNT → Response Category
    # --------------------------------------------------
    # Medical logic:
    # Number of follicles indicates ovarian response

    df["Follicle_Response"] = pd.cut(
        df["Follicle_Count"],
        bins=[0, 3, 8, 20, 100],
        labels=["Poor", "Normal", "Good", "High"]
    )

    # --------------------------------------------------
    # ESTRADIOL → Normalized & Threshold Features
    # --------------------------------------------------
    # Medical logic:
    # Absolute E2 alone is misleading; context matters

    # Estradiol per follicle (normalization)
    df["E2_per_Follicle"] = (
        df["Estradiol_pg_mL"] / (df["Follicle_Count"] + 1)
    )

    # High estradiol clinical flag
    df["High_E2"] = (df["Estradiol_pg_mL"] > 1000).astype(int)

    # --------------------------------------------------
    # PROGESTERONE → Risk Flag
    # --------------------------------------------------
    # Medical logic:
    # Premature progesterone rise affects trigger timing

    df["High_Progesterone"] = (
        df["Progesterone_ng_mL"] > 1
    ).astype(int)

    # --------------------------------------------------
    # HORMONE INTERACTION FEATURES
    # --------------------------------------------------
    # Medical logic:
    # Trigger decision depends on hormone balance, not isolation

    # Estradiol to Progesterone ratio
    df["Hormone_Ratio"] = (
        df["Estradiol_pg_mL"] / (df["Progesterone_ng_mL"] + 0.01)
    )

    # --------------------------------------------------
    # CLEANUP 
    # --------------------------------------------------
    # Keep raw features + engineered features
    # Categorical features will be one-hot encoded later

    return df
