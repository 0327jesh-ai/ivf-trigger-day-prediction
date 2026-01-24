# ==================================================
# ML Training Script – IVF Trigger Day Prediction
# ==================================================
# Safe version: Automatically resolves project root
# and ensures imports work from anywhere
# ==================================================

# =============================
# IMPORTS
# =============================

# System Libraries
import os
import sys
import pathlib

# -----------------------------
# Resolve project root dynamically
# -----------------------------
# This ensures the script can be run from ANY directory
# and still find 'features' and 'Database' folders.

# __file__ is the current script path
current_file = pathlib.Path(__file__).resolve()

# Assume project structure:
# ML_Trigger/
# ├─ features/
# ├─ Database/
# └─ ML_Model/train_model.py

project_root = current_file.parent.parent.resolve()
print(f"Project root detected: {project_root}")

# Add project root to sys.path if not already added
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path")

# =============================
# DATA PROCESSING LIBRARIES
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =============================
# MODEL MANAGEMENT LIBRARIES
# =============================
import joblib
import mlflow
import mlflow.sklearn

# =============================
# CUSTOM MODULE IMPORTS
# =============================
# Now Python can find these modules no matter the cwd
from features.feature_engineering import apply_feature_engineering
from Database.postgres import fetch_trigger_day_data

# =============================
# REST OF YOUR SCRIPT REMAINS THE SAME
# =============================

# Example: Load data
df = fetch_trigger_day_data(table_name="ivf_data")
print(f"Data loaded: {df.shape}")

# Apply feature engineering
df = apply_feature_engineering(df)
print(f"Features processed: {df.shape}")
