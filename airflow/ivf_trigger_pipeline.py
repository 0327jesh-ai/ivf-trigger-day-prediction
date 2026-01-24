# ============================================================
# IVF Trigger Prediction – Airflow DAG (Optimized & Commented)
# ============================================================
# Orchestrates the ML pipeline: Ingestion -> Validation ->
# Database load -> Model training. Designed for clarity and
# safe execution within Airflow.
# ============================================================

# -----------------------------
# Airflow & Operator Imports
# -----------------------------
from airflow import DAG                                      # Core DAG object and context manager
from airflow.providers.standard.operators.python import PythonOperator  # Run Python callables as tasks

# -----------------------------
# Standard Library Imports
# -----------------------------
from datetime import datetime                                # DAG start date & scheduling
import subprocess                                            # Spawn external processes (run scripts)
import sys                                                   # Access Python executable used by Airflow
import logging                                               # Structured logging for task execution
import os                                                    # Path operations and file joins

# -----------------------------
# Configuration / Constants
# -----------------------------
# Project root on the filesystem (used as working directory when running scripts)
PROJECT_ROOT = "J:/ML_Trigger"

# Configure basic logging (Airflow still captures task output; this is for clarity)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# Helper Functions
# -----------------------------
def run_script(script_relative_path):
    """
    Execute a Python script using Airflow's Python interpreter.

    - script_relative_path: path to the script relative to PROJECT_ROOT.
    - Uses subprocess.run with check=True so Airflow task fails on error.
    - Logs start/finish and ensures cwd is PROJECT_ROOT so scripts use project-relative paths.
    """
    script_path = os.path.join(PROJECT_ROOT, script_relative_path)
    logging.info("Starting script: %s", script_path)
    try:
        subprocess.run([sys.executable, script_path], check=True, cwd=PROJECT_ROOT)
        logging.info("Finished script: %s", script_path)
    except subprocess.CalledProcessError as exc:
        logging.error("Script failed: %s (returncode=%s)", script_path, getattr(exc, "returncode", None))
        raise

# -----------------------------
# Default DAG Arguments
# -----------------------------
default_args = {
    "owner": "ivf",                      # Responsible party for the DAG
    "start_date": datetime(2024, 1, 1),  # Logical start date (useful for scheduling)
    "retries": 1,                        # Number of retries on task failure
}

# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id="ivf_trigger_pipeline",      # Unique DAG identifier
    default_args=default_args,
    schedule_interval=None,             # Manual trigger only (set a cron or preset if scheduling desired)
    catchup=False,                      # Do not backfill missed runs
    tags=["ivf", "ml_pipeline"],
    max_active_runs=1,                  # Only one concurrent run to avoid race conditions
) as dag:

    # -----------------------------
    # Ingestion Task
    # -----------------------------
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=run_script,
        op_args=["Ingestion/ingest_data.py"],
        dag=dag,
    )

    # -----------------------------
    # Validation Task
    # -----------------------------
    validate = PythonOperator(
        task_id="validation",
        python_callable=run_script,
        op_args=["Validation/validate_with_ge.py"],
        dag=dag,
    )

    # -----------------------------
    # Push to Database Task
    # -----------------------------
    push_db = PythonOperator(
        task_id="postgres",
        python_callable=run_script,
        op_args=["Database/postgres.py"],
        dag=dag,
    )

    # -----------------------------
    # Model Training Task
    # -----------------------------
    train = PythonOperator(
        task_id="train_model",
        python_callable=run_script,
        op_args=["ML_Model/train_model.py"],
        dag=dag,
    )

    # -----------------------------
    # Task Dependency Graph (linear pipeline)
    # -----------------------------
    ingest >> validate >> push_db >> train

    # -----------------------------
    # Optional: Example of parallel tasks (commented)
    # -----------------------------
    # If you later add independent steps (e.g., reporting), they can run in parallel:
    # report = PythonOperator(
    #     task_id="generate_report",
    #     python_callable=run_script,
    #     op_args=["Reports/generate_report.py"],
    #     dag=dag,
    # )
    # validate >> [push_db, report]