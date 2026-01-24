# ==================================================
# DATA DRIFT MONITORING – IVF TRIGGER PREDICTION
# ==================================================
# Purpose:
# - Detect feature-level data drift
# - Compare reference vs current datasets
# - Generate HTML drift report
# ==================================================

# -----------------------------
# Core Libraries
# -----------------------------
import pandas as pd                     # Data handling
import pathlib                          # Path resolution
import sys                              # Import path handling

# -----------------------------
# Evidently (STABLE API)
# -----------------------------
from evidently.report import Report     # Main report engine
from evidently.metrics import DataDriftTable  # Drift metric

# -----------------------------
# Project Root Setup
# -----------------------------
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# -----------------------------
# Feature Engineering
# -----------------------------
from features.feature_engineering import apply_feature_engineering

# -----------------------------
# Load Reference Data
# -----------------------------
reference = pd.read_csv("Monitoring/reference_data.csv")
reference = apply_feature_engineering(reference)

# -----------------------------
# Load Current Data
# -----------------------------
current = pd.read_csv("data/Trigger_day_prediction.csv")
current = apply_feature_engineering(current)

# -----------------------------
# Create Drift Report
# -----------------------------
report = Report(
    metrics=[
        DataDriftTable()               # Feature-wise drift summary
    ]
)

# -----------------------------
# Run Drift Calculation
# -----------------------------
report.run(
    reference_data=reference,
    current_data=current
)

# -----------------------------
# Save Report
# -----------------------------
report.save_html("Monitoring/drift_report.html")

print("Drift report generated successfully")
