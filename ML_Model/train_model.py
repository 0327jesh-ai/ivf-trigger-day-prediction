# ==================================================
# ML Training Script – IVF Trigger Day Prediction
# ==================================================
# Purpose:
# - Load raw IVF clinical data from PostgreSQL
# - Apply feature engineering
# - Preprocess data (scaling & encoding)
# - Perform unsupervised clustering (KMeans)
# - Train Random Forest classifier
# - Evaluate performance
# - Track everything using MLflow
# - Save models for deployment
# ==================================================

# =============================
# IMPORTS
# =============================

# Data Processing & Analysis
import pandas as pd                              # DataFrames and data manipulation utilities
import os                                        # Operating system interactions (directories, files)
import sys                                       # System-specific parameters and path operations
import pathlib                                   # Object-oriented filesystem path handling
import joblib                                    # Model serialization and deserialization

# Machine Learning Libraries
from sklearn.model_selection import train_test_split  # Split data into train/test subsets
from sklearn.ensemble import RandomForestClassifier   # Random Forest classifier algorithm
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics

# Experiment Tracking & MLflow
import mlflow                                    # Experiment tracking and metrics logging
import mlflow.sklearn                           # MLflow scikit-learn integration

# =============================
# PROJECT PATH SETUP
# =============================
# Purpose: Establish absolute path to project root for consistent module imports
# __file__ gives current script location, .parent.parent navigates to project root
# sys.path.append allows importing custom modules from project structure

project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# =============================
# CUSTOM MODULE IMPORTS
# =============================
# Import custom pipeline functions for ML workflow

from Features.feature_engineering import apply_feature_engineering  # Transforms raw data into ML features
from Features.preprocessing import apply_preprocessing               # Scales & encodes features
from Clustering.patient_clustering import apply_clustering          # KMeans clustering for patient segmentation
from Database.postgres import fetch_trigger_day_data, save_dataframe_to_postgres  # Database I/O operations
from Features.eda import run_eda_ivf   # or wherever you saved the function

# =============================
# STEP 1: LOAD DATA FROM DATABASE
# =============================
# Purpose: Retrieve raw IVF patient data from PostgreSQL database
# This data contains all clinical measurements and outcomes

print("\n" + "=" * 60)
print("STEP 1: LOADING DATA FROM POSTGRESQL")
print("=" * 60)

# Connect to PostgreSQL and fetch raw patient data
# Returns pandas DataFrame with all patient records from ivf_data table
df = fetch_trigger_day_data(table_name="ivf_data")
print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================
# step 1.5 EXPLORATORY DATA ANALYSIS (EDA)
# =============================
# Purpose:
# - Understand raw data quality
# - Detect missing values, outliers, correlations
# - Generate audit artifacts BEFORE any transformation
# - Prevent silent data issues from propagating downstream

eda_results = run_eda_ivf(
    df,
    target_col="Trigger_Recommended (0/1)",
    save_reports=True,
    show_plots=False   # Set to True to visualize plots
)


# =============================
# STEP 2: FEATURE ENGINEERING
# =============================
# Purpose: Transform raw clinical measurements into ML-ready features
# Examples: Age normalization, hormone level scaling, binary encoding

print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# Apply feature engineering transformations to raw data
# Creates derived features that improve model learning and prediction
df = apply_feature_engineering(df)
print(f"✓ Features created: {df.shape[0]} rows × {df.shape[1]} columns")

# Save feature-engineered data for reproducibility and audit trail
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/Trigger_day_prediction_processed.csv", index=False)
print("✓ Feature-engineered data saved: data/processed/Trigger_day_prediction_processed.csv")

# =============================
# STEP 3: PREPARE FEATURES & TARGET
# =============================
# Purpose: Separate input features (X) from target variable (y)
# X: Features used for model input during prediction
# y: Target variable (0/1) that model learns to predict

# Define target column name
# Values: 0 = no trigger recommended, 1 = trigger recommended on specific day
TARGET = "Trigger_Recommended (0/1)"

# Separate features from target
X = df.drop(columns=[TARGET])  # All columns except target (input features)
y = df[TARGET]                  # Target column only (prediction target)

# =============================
# STEP 4: TRAIN-TEST SPLIT
# =============================
# Purpose: Divide data into training (80%) and testing (20%) subsets
# Training set: Used to fit/train the model
# Testing set: Used to evaluate model performance on unseen data
# stratify=y ensures class distribution (0/1 ratio) is maintained in both splits

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,              # 20% of data for testing, 80% for training
    random_state=42,            # Fixed seed for reproducible splits
    stratify=y                  # Maintain same class ratio in train and test
)

print(f"✓ Train samples: {X_train.shape[0]} ({100*X_train.shape[0]/len(X):.1f}%)")
print(f"✓ Test samples: {X_test.shape[0]} ({100*X_test.shape[0]/len(X):.1f}%)")

# =============================
# START MLFLOW EXPERIMENT TRACKING
# =============================
# Purpose: Track all metrics, parameters, and artifacts for this training run
# MLflow enables reproducibility, comparison across runs, and model versioning
# All operations inside this context will be logged to MLflow backend

mlflow.set_experiment("ivf_trigger_model")  # Create/select MLflow experiment

with mlflow.start_run(run_name="ivf_training_with_clustering"):  # Begin tracking this specific run

    # =============================
    # STEP 5: PREPROCESSING (SCALING & ENCODING)
    # =============================
    # Purpose: Normalize numeric features and encode categorical variables
    # Scaling ensures all features are on same numeric range (important for tree-based models)
    # Fit scaler on training data only, then apply to test data (prevents data leakage)

    print("\n" + "=" * 60)
    print("STEP 5: PREPROCESSING (SCALING & ENCODING)")
    print("=" * 60)

    # Apply preprocessing to training set
    # mode="train": Fit StandardScaler and OneHotEncoder on training data, save fitted transformers
    X_train = apply_preprocessing(X_train, mode="train")
    
    # Apply preprocessing to test set
    # mode="inference": Load saved transformers and apply to test data (ensure consistency)
    X_test = apply_preprocessing(X_test, mode="inference")
    print(f"✓ Preprocessing completed: Train {X_train.shape}, Test {X_test.shape}")

    # =============================
    # STEP 6: UNSUPERVISED CLUSTERING (KMEANS PATIENT SEGMENTATION)
    # =============================
    # Purpose: Segment patients into homogeneous groups using KMeans algorithm
    # Clustering identifies patient subpopulations with different clinical patterns
    # Cluster assignment becomes a new categorical feature for the model
    # Train clustering on training set ONLY to avoid data leakage

    print("\n" + "=" * 60)
    print("STEP 6: PATIENT CLUSTERING (KMEANS)")
    print("=" * 60)

    # Train clustering model on training data only
    # apply_clustering() performs grid search for optimal K using silhouette score
    # Automatically saves trained model to 'models/patient_kmeans.pkl'
    # Returns training data with new 'patient_cluster' column
    X_train = apply_clustering(X_train)
    print(f"✓ Clustering applied to training data: {X_train.shape[1]} features (including cluster)")

    # Apply same clustering model to test data
    # Load pre-trained model fitted on training data
    # Ensures test data is assigned using same cluster centroids (maintains consistency)
    # Prevents test data from influencing cluster definitions (no data leakage)
    cluster_model = joblib.load("models/patient_kmeans.pkl")
    X_test["patient_cluster"] = cluster_model.predict(X_test)
    print(f"✓ Clustering applied to test data: {X_test.shape[1]} features (including cluster)")

    # =============================
    # STEP 7: SAVE PREPROCESSED DATA TO POSTGRESQL
    # =============================
    # Purpose: Persist preprocessed training and test datasets
    # Enables audit trails, debugging, and data drift analysis
    # Separate tables for train/test maintain data integrity

    print("\n" + "=" * 60)
    print("STEP 7: SAVING PREPROCESSED DATA TO POSTGRESQL")
    print("=" * 60)

    # Reattach target column to training features
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    # Reattach target column to test features
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Save preprocessed training data to PostgreSQL
    # if_exists="replace" regenerates table for each training run (fresh data)
    save_dataframe_to_postgres(
        df=train_df,
        table_name="ivf_preprocessed_train",
        if_exists="replace"
    )

    # Save preprocessed test data to PostgreSQL
    save_dataframe_to_postgres(
        df=test_df,
        table_name="ivf_preprocessed_test",
        if_exists="replace"
    )
    
    print("✓ Preprocessed datasets saved to PostgreSQL")

    # =============================
    # STEP 8: TRAIN RANDOM FOREST CLASSIFIER
    # =============================
    # Purpose: Train ensemble model that combines multiple decision trees
    # Random Forest advantages: Handles non-linearity, robust to outliers, provides feature importance
    # n_estimators=100 provides good balance between accuracy and training time

    print("\n" + "=" * 60)
    print("STEP 8: TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # Initialize Random Forest with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,           # Number of decision trees in the ensemble
        random_state=42,            # Fixed seed for reproducible results
        n_jobs=-1                   # Use all available CPU cores for parallel training
    )

    # Fit model to training data
    # Learns patterns and decision boundaries from preprocessed training features
    # y_train: Binary target (0 or 1 - trigger decision)
    print("Training in progress...")
    model.fit(X_train, y_train)
    print("✓ Model training completed")

    # =============================
    # STEP 9: MODEL EVALUATION ON TEST SET
    # =============================
    # Purpose: Assess model performance on unseen test data
    # Test data represents new patients the model has never seen during training
    # Evaluating on test data gives realistic estimate of production performance

    print("\n" + "=" * 60)
    print("STEP 9: MODEL EVALUATION")
    print("=" * 60)

    # Generate predictions on test set
    # Returns array of predicted labels (0 or 1) for each test sample
    y_pred = model.predict(X_test)
    
    # Calculate overall accuracy
    # Accuracy = (Correct Predictions) / (Total Predictions)
    # Range: 0.0 to 1.0 (0% to 100%)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy on Test Set: {acc:.4f} ({100*acc:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # =============================
    # STEP 10: LOG RESULTS TO MLFLOW
    # =============================
    # Purpose: Track model parameters, metrics, and artifacts for reproducibility and comparison
    # MLflow enables viewing model performance over time and comparing across training runs

    print("\n" + "=" * 60)
    print("STEP 10: LOGGING TO MLFLOW")
    print("=" * 60)

    # Log hyperparameters (configuration inputs used during training)
    mlflow.log_param("model", "RandomForest")       # Model type
    mlflow.log_param("n_estimators", 100)           # Number of trees
    mlflow.log_param("test_split", 0.2)             # Data split ratio (20% test, 80% train)

    # Log performance metrics (evaluation results on test data)
    mlflow.log_metric("accuracy", acc)              # Overall accuracy score

    # Log trained model artifact to MLflow backend
    # Enables model versioning, reproducibility, and production deployment
    mlflow.sklearn.log_model(model, artifact_path="rf_model")
    
    print(f"✓ Metrics and model logged to MLflow")

    # =============================
    # STEP 11: SAVE TRAINED MODEL FOR DEPLOYMENT
    # =============================
    # Purpose: Persist model to disk for deployment to API/batch services
    # joblib: Efficient serialization format optimized for scikit-learn models

    print("\n" + "=" * 60)
    print("STEP 11: SAVE MODEL FOR DEPLOYMENT")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Serialize and save trained model using joblib
    # Allows model to be loaded later without retraining
    model_path = "models/rf_model.pkl"
    joblib.dump(model, model_path)
    
    # Get file size for logging
    model_size = os.path.getsize(model_path) / 1024
    print(f"✓ Model saved to {model_path}")
    print(f"  File size: {model_size:.2f} KB")

# =============================
# STEP 12: TRAINING COMPLETION SUMMARY
# =============================
# Final summary of training results and next steps for deployment

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\n Training Summary:")
print(f"  Final Accuracy: {acc:.4f} ({100*acc:.2f}%)")
print(f"  Training Samples: {X_train.shape[0]}")
print(f"  Test Samples: {X_test.shape[0]}")
print(f"  Total Features (including cluster): {X_train.shape[1]}")

print("\n Next Steps:")
print("  1. View MLflow Results: mlflow ui --port 5050")
print("     → Compare metrics across training runs")
print("  2. Deploy FastAPI Server: python Api/main.py")
print("     → Start REST API service for real-time predictions")
print("  3. Monitor Data Drift: python Monitoring/data_drift.py")
print("     → Check if production data changed from training distribution")
print("  4. Validate with Great Expectations: python Validation/validate_with_ge.py")
print("     → Run data quality checks on input data")

print("\n" + "=" * 60 + "\n")
