# -----------------------------
# PUSH DATA TO POSTGRESQL
# -----------------------------
# Purpose:
# - Upload validated DataFrame into PostgreSQL
# - Create / replace table automatically
# - Ensure secure handling of credentials
# -----------------------------

import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


def push_trigger_day_data(df: pd.DataFrame):
    """
    Upload validated Trigger Day DataFrame to PostgreSQL
    """

    # ------------------------------------------------
    # 1. POSTGRESQL CONNECTION PARAMETERS
    # ------------------------------------------------
    username = "postgres"

    # Password encoded to handle special characters like @, #, etc.
    password = quote_plus("@Gorgeous2703")

    host = "localhost"
    port = "5432"
    database = "ivf"

    # ------------------------------------------------
    # 2. CREATE SQLALCHEMY ENGINE
    # ------------------------------------------------
    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    )

    print("Connected to PostgreSQL")

    # ------------------------------------------------
    # 3. PUSH DATAFRAME TO DATABASE
    # ------------------------------------------------
    # if_exists="replace" → drops table and recreates it
    # index=False → avoids extra index column
    df.to_sql(
        name="ivf_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("CSV uploaded to PostgreSQL successfully!")


# ------------------------------------------------
# RUN DIRECTLY FOR TESTING
# ------------------------------------------------
if __name__ == "__main__":

    # Example usage (only for testing)
    sample_df = pd.read_csv("data/Trigger_day_prediction.csv")
    push_trigger_day_data(sample_df)
