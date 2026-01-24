# ============================================================
# IMPORTS
# ============================================================
# pandas: Data manipulation and DataFrame operations
import pandas as pd
# joblib: Serialization/deserialization of Python objects (scaler)
import joblib
# os: File and directory operations
import os
# StandardScaler: Normalize numeric features to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION - FILE PATHS
# ============================================================
# Directory where artifacts (scalers, models) are stored
# Using os.path.join for cross-platform compatibility (Windows/Linux/Mac)
ARTIFACTS_DIR = "artifacts"
# Full path to the saved StandardScaler object used for normalization
# Scaler is trained on training data and loaded for inference/batch processing
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")

# ============================================================
# CONFIGURATION - FEATURE COLUMNS
# ============================================================
# Numeric columns that require scaling using StandardScaler
# These features have varying ranges and need normalization (zero mean, unit variance)
# Scaling ensures no single feature dominates the model due to larger magnitude
NUMERIC_SCALE_COLS = [
    "Age",
    "AMH (ng/mL)",
    "Day",
    "Follicle_Count",
    "Estradiol_pg_mL",
    "Progesterone_ng_mL",
    "E2_per_Follicle",
    "Hormone_Ratio"
]

# Categorical columns that require one-hot encoding
# These features have discrete categorical values that need to be converted to numeric
# One-hot encoding creates binary columns for each category
CATEGORICAL_COLS = [
    "Age_Group",
    "AMH_Level",
    "Day_Zone",
    "Follicle_Response"
]


def apply_preprocessing(df: pd.DataFrame, mode: str = "inference") -> pd.DataFrame:
    """
    Apply preprocessing steps consistently for training and inference pipelines.

    This function performs two critical preprocessing operations:
    1. One-hot encode categorical variables to convert them into numeric format
    2. Scale numeric features using StandardScaler for consistent feature ranges

    Parameters:
        df (pd.DataFrame): Feature-engineered dataset from feature_engineering module
        mode (str): Processing mode - "train" or "inference"
                   "train": Fit StandardScaler on data and save to disk
                   "inference": Load pre-trained scaler and apply transformation
                   Default is "inference"

    Returns:
        pd.DataFrame: Preprocessed dataset with:
                     - One-hot encoded categorical features
                     - Scaled numeric features
                     - Ready for model input

    Raises:
        FileNotFoundError: If mode="inference" but SCALER_PATH does not exist.
                          This happens if training has not been run yet.

    Example:
        # Training: fit and save scaler
        df_train = apply_preprocessing(df_train, mode="train")

        # Inference: load and apply saved scaler
        df_test = apply_preprocessing(df_test, mode="inference")
    """

    # --------------------------------------------------
    # STEP 1: ONE-HOT ENCODING
    # --------------------------------------------------
    # Convert categorical variables into numeric format using one-hot encoding
    # This transformation creates binary (0/1) columns for each category
    # drop_first=True removes the first category for each feature
    # This avoids the "dummy variable trap" (multicollinearity) in linear models
    # Example: "Age_Group" → Age_Group_30-35, Age_Group_35-40, etc.
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True,
        dtype=int  # <-- ensures 0/1 instead of True/False
    )

    # --------------------------------------------------
    # STEP 2: SCALING (STANDARDIZATION)
    # --------------------------------------------------
    # Scale numeric features to have zero mean (center) and unit variance (normalize)
    # This ensures all numeric features are on the same scale
    # Important for distance-based and gradient-based algorithms
    # StandardScaler: (x - mean) / std_dev
    
    if mode == "train":
        # TRAINING MODE: Fit scaler on training data and save to disk
        # This mode should only be used during model training
        
        # Create new StandardScaler instance
        # Will learn mean and standard deviation from training data
        scaler = StandardScaler()

        # Fit the scaler to training data
        # Calculates mean and standard deviation for each numeric column
        # These statistics are stored in the scaler object
        scaler.fit(df[NUMERIC_SCALE_COLS])

        # Create artifacts directory if it doesn't already exist
        # This directory stores all model artifacts (scaler, models, etc.)
        # exist_ok=True prevents error if directory already exists
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # Save the fitted scaler to disk using joblib
        # Joblib is optimized for saving large Python objects and scikit-learn models
        # Allows the scaler to be loaded later for consistent preprocessing
        joblib.dump(scaler, SCALER_PATH)

    else:
        # INFERENCE MODE: Load pre-trained scaler and apply to new data
        # This mode should be used for batch predictions, API requests, production inference
        # Uses the same scaler from training to ensure consistency
        
        # Check if scaler file exists before attempting to load
        # Raises informative error if training hasn't been run yet
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                "Scaler not found. Train the model first to generate scaler."
            )

        # Load the pre-trained scaler from disk
        # This scaler has already learned mean/std from training data
        scaler = joblib.load(SCALER_PATH)

    # --------------------------------------------------
    # STEP 3: APPLY TRANSFORMATION
    # --------------------------------------------------
    # Apply the scaler transformation to all numeric columns
    # Both training and inference modes use the same scaler object at this point
    # In training: scaler was just fit, so this applies transformations to training data
    # In inference: scaler was loaded, so this applies transformations to new data
    # Result: All numeric features are standardized (mean=0, std=1)
    df[NUMERIC_SCALE_COLS] = scaler.transform(
        df[NUMERIC_SCALE_COLS]
    )

    # Return the fully preprocessed dataframe
    # Ready for input to machine learning model
    return df
