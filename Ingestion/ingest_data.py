# -----------------------------
#  DATA INGESTION
# -----------------------------
# Purpose:
# - Read the Trigger Day Prediction CSV file
# - Load it into a Pandas DataFrame
# - This is the FIRST step of the data pipeline
# -----------------------------

import pandas as pd  # Pandas is used for data reading and manipulation


def ingest_trigger_day_data():
    """
    This function performs data ingestion.
    
    What it does:
    1. Reads the CSV file from the data folder
    2. Loads data into a Pandas DataFrame
    3. Prints a success message
    4. Displays first 5 rows (sanity check)
    5. Returns the DataFrame for next steps
    """

    # Read CSV file into DataFrame
    df = pd.read_csv("data/Trigger_day_prediction.csv")

    # Confirmation message
    print("Trigger Day Prediction data ingested successfully")

    # Display first 5 rows to verify data
    print("Sample data:")
    print(df.head())

    # Return DataFrame for validation / database steps
    return df


# This block ensures the function runs
# ONLY when this file is executed directly
# (not when imported into Airflow or other scripts)
if __name__ == "__main__":
    ingest_trigger_day_data()
