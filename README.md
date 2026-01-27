# README.md

# IVF Trigger Day Prediction

## Project Overview

End-to-end machine learning pipeline to predict the optimal IVF trigger day using patient clinical and hormonal data. The system automates data ingestion, validation, preprocessing, model training, experiment tracking, and deployment via a REST API.

## Key Features

* **Data Ingestion:** Reads raw CSV data into pandas DataFrames.
* **Data Validation:** Ensures schema, quality, and distribution integrity using Great Expectations.
* **Database Integration:** Stores raw and processed data in PostgreSQL.
* **Feature Engineering:** Generates clinically meaningful features for model input.
* **Clustering:** Segments patients using KMeans for better prediction.
* **Model Training:** Random Forest classifier with train/test split.
* **Experiment Tracking:** Logs metrics and artifacts using MLflow.
* **Orchestration:** Apache Airflow DAG automates pipeline steps.
* **Deployment:** FastAPI REST API for real-time predictions.

## Folder Structure

```
ML_Trigger/
├── Ingestion/                # Data ingestion scripts
├── Validation/               # Data validation scripts using Great Expectations
├── Database/                 # PostgreSQL interface scripts
├── Features/                 # Feature engineering and preprocessing
├── Clustering/               # KMeans clustering scripts
├── ML_Model/                 # Model training scripts
├── api/                      # FastAPI deployment code
├── dags/                     # Airflow DAG definition
├── data/                     # Raw and processed datasets
├── models/                   # Saved model artifacts
├── artifacts/                # Scalers and other artifacts
├── requirements.txt          # Project dependencies
└── README.md                 # Project overview and instructions
```

## How to Run

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Run Airflow DAG

* Place DAG file in `dags/` folder.
* Start Airflow webserver and scheduler.
* Trigger `ivf_trigger_pipeline` manually or via schedule.

### 3. Training

* Trigger DAG will execute scripts:

  1. Ingestion
  2. Validation
  3. Database load
  4. Feature engineering + preprocessing
  5. Clustering + model training
* Metrics and model artifacts logged to MLflow.

### 4. Deployment

```bash
uvicorn api.main:app --reload
```

* API Root: `http://127.0.0.1:8000/`
* Prediction Endpoint: `http://127.0.0.1:8000/predict`

### 5. Monitoring & Validation

* Data drift detection via `Monitoring/data_drift.py`
* Great Expectations validation for new input data.
* MLflow tracks model performance over time.

## Tech Stack

* Python, Pandas, NumPy
* Scikit-learn, Joblib
* PostgreSQL, SQLAlchemy
* Apache Airflow
* MLflow
* Great Expectations, Evidently
* FastAPI, Uvicorn

## Notes

* Project is modular: working code (training pipeline) is separate from deployment code (FastAPI).
* Deployment is lightweight; training is orchestrated and automated.
* Designed for reproducibility and MLOps best practices.

## Author

Jeshwanth
