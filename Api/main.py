# ==================================================
# api/main.py – Trigger Prediction API
# ==================================================

# -----------------------------
# Libraries
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
import pathlib

# -----------------------------
# Project Paths
# -----------------------------
# Automatically detect project root to safely locate model
project_root = pathlib.Path(__file__).parent.parent.resolve()
model_path = project_root / "models" / "rf_model.pkl"

# -----------------------------
# Load trained model
# -----------------------------
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")
model = joblib.load(model_path)

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Trigger Prediction API")

# -----------------------------
# Request schema
# -----------------------------
class PredictionRequest(BaseModel):
    """
    Expected JSON body for prediction.
    data: list of dictionaries, each representing one patient's features.
    """
    data: List[dict]

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    """
    Simple API health check.
    """
    return {"message": "Trigger Prediction API is live"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    """
    Receives JSON input, aligns it with model features, and returns predictions.
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(req.data)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Align input columns with model training features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]  # Reorder columns exactly

    # Make predictions
    preds = model.predict(df)

    # Return results
    return {"predictions": preds.tolist()}

# ==================================================
# HOW TO RUN THIS API
# ==================================================
"""
1. Go to project root:
   cd /mnt/j/ML_Trigger       # WSL/Linux
   or
   cd J:\ML_Trigger           # Windows

2. Start the API:
   uvicorn api.main:app --reload

3. Open in browser:
   - API root: http://127.0.0.1:8000/
   - Docs: http://127.0.0.1:8000/docs

4. Test /predict endpoint by sending JSON:
   {
       "data": [
           {"Age": 30, "AMH_Level": 2.1, "BMI": 24, "Other_Feature": 1}
       ]
   }

5. Next:
   - Connect API to dashboard or frontend
   - Update model when new data is available
"""
# ==================================================
