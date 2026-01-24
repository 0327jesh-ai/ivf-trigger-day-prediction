# --------------------------------------------------
# Save Reference Data for Monitoring
# --------------------------------------------------
# Purpose:
# - Store feature-engineered training data
# - Used as baseline for data drift detection
# --------------------------------------------------

import pandas as pd
import sys
import pathlib

# --------------------------------------------------
# Add project root to path
# --------------------------------------------------
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# --------------------------------------------------
# Import feature engineering (single source of truth)
# --------------------------------------------------
from features.feature_engineering import apply_feature_engineering

# --------------------------------------------------
# Load raw training data
# --------------------------------------------------
RAW_DATA_PATH = "data/Trigger_day_prediction.csv"
REFERENCE_DATA_PATH = "Monitoring/reference_data.csv"

df = pd.read_csv(RAW_DATA_PATH)
print("Raw data loaded:", df.shape)

# --------------------------------------------------
# Apply feature engineering
# --------------------------------------------------
df_fe = apply_feature_engineering(df)
print("Feature engineering applied:", df_fe.shape)

# --------------------------------------------------
# Save reference dataset
# --------------------------------------------------
df_fe.to_csv(REFERENCE_DATA_PATH, index=False)
print(f"Reference data saved to {REFERENCE_DATA_PATH}")
