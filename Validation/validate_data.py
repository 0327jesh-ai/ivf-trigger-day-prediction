# -----------------------------
# DATA VALIDATION
# -----------------------------
# Purpose:
# - Ensure data quality before ML / database
# - Catch missing values & impossible values
# - Stop pipeline if validation fails
# -----------------------------

import pandas as pd
import os


def validate_trigger_day_data():
    """
    Validates Trigger Day Prediction dataset
    using explicit pandas checks (stable & production-safe)
    """

    # ------------------------------------------------
    # GET PROJECT ROOT DIRECTORY
    # ------------------------------------------------
    # Moves one level up from Validation folder
    project_root = os.path.dirname(os.path.dirname(__file__))

    # ------------------------------------------------
    # CSV FILE PATH
    # ------------------------------------------------
    csv_path = os.path.join(
        project_root,
        "data",
        "Trigger_day_prediction.csv"
    )

    # ------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------
    df = pd.read_csv(csv_path)

    print("Dataset loaded successfully")
    print("Columns found:")
    print(df.columns.tolist())

    # ------------------------------------------------
    # BASIC DATA QUALITY CHECKS
    # ------------------------------------------------

    # Dataset must not be empty
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty")

    # Patient ID must not be null
    if df["Patient_ID"].isnull().any():
        raise ValueError("Null values found in Patient_ID")

    # Age should be realistic for IVF patients
    if not df["Age"].between(18, 50).all():
        raise ValueError("Invalid Age values detected")

    # AMH should be non-negative
    if (df["AMH (ng/mL)"] < 0).any():
        raise ValueError("Negative AMH values detected")

    # Day of cycle should be reasonable
    if not df["Day"].between(1, 20).all():
        raise ValueError("Invalid Day values detected")

    # Trigger recommendation should be binary
    if not df["Trigger_Recommended (0/1)"].isin([0, 1]).all():
        raise ValueError("Invalid Trigger_Recommended values detected")

    # ------------------------------------------------
    # VALIDATION SUCCESS
    # ------------------------------------------------
    print(" Data validation passed successfully")


if __name__ == "__main__":
    validate_trigger_day_data()
