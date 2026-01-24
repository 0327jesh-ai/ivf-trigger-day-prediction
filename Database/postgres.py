# ============================================================
# PostgreSQL DATABASE MODULE - DATA PERSISTENCE & RETRIEVAL
# ============================================================
# Purpose:
# - Upload validated pandas DataFrames into PostgreSQL tables
# - Fetch data from PostgreSQL tables into pandas DataFrames
# - Create or replace tables automatically based on if_exists parameter
# - Ensure secure and robust database connection handling
#
# Module Flow:
# 1. Define database connection credentials (localhost)
# 2. Create SQLAlchemy engine for connection pooling
# 3. Upload/Download data using pandas to_sql() and read_sql()
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
# pandas: DataFrames for data manipulation and SQL operations
import pandas as pd
# sqlalchemy: ORM and database connection management
from sqlalchemy import create_engine
# urllib.parse: URL encode special characters in database passwords
from urllib.parse import quote_plus


# ============================================================
# DATABASE CONFIGURATION
# ============================================================
# Database connection parameters
# NOTE: In production, use environment variables or config files
# instead of hardcoded credentials for security
DB_CONFIG = {
    "username": "postgres",              # PostgreSQL default superuser
    "password": "@Gorgeous2703",         # NOTE: Store in .env or secrets manager in production
    "host": "localhost",                 # Local database server
    "port": "5432",                      # PostgreSQL default port
    "database": "ivf"                    # Target database name
}


def _get_db_engine():
    """
    Create and return a SQLAlchemy database engine.
    
    Internal helper function to avoid code duplication across multiple functions.
    Encodes password to handle special characters safely in connection URL.
    
    Returns:
        sqlalchemy.engine.Engine: Database engine for executing SQL queries
        
    Raises:
        Exception: If database connection fails
    """
    # Encode password to handle special characters like @, #, $, etc.
    # URL encoding converts these to safe URL-compatible format
    encoded_password = quote_plus(DB_CONFIG["password"])
    
    # Create connection string using PostgreSQL dialect + psycopg2 driver
    # Format: postgresql+psycopg2://user:password@host:port/database
    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_CONFIG['username']}:{encoded_password}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/"
        f"{DB_CONFIG['database']}"
    )
    
    # Create SQLAlchemy engine
    # Engine handles connection pooling and query execution
    engine = create_engine(connection_string)
    
    return engine



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

# ============================================================
# SAVE DATAFRAME TO POSTGRES (FLEXIBLE TABLE HANDLING)
# ============================================================

def save_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append"
):
    """
    Save pandas DataFrame to a specified PostgreSQL table with flexible handling modes.
    
    This function provides a convenient way to persist DataFrames to PostgreSQL
    with different strategies for handling existing tables. Useful for saving
    preprocessed data, validation datasets, or intermediate results.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save to PostgreSQL database
                          Each column becomes a table column with inferred type
        table_name (str): Target table name in PostgreSQL
                         If table doesn't exist, it will be created automatically
        if_exists (str): How to behave if table already exists
                        - "append" (default): Insert rows into existing table
                        - "replace": Drop existing table and create new one
                        - "fail": Raise error if table exists
    
    Returns:
        None - Function only prints confirmation messages
    
    Raises:
        Exception: If database connection fails or table operation encounters error
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        >>> save_dataframe_to_postgres(df, "users", if_exists="replace")
        ✓ Data saved to PostgreSQL table: users
          Rows: 2 | Columns: 2 | Mode: replace
    """
    
    # ================================================
    # STEP 1: ESTABLISH DATABASE CONNECTION
    # ================================================
    # Call helper function to get SQLAlchemy engine
    # This ensures consistent connection parameters across all database functions
    # Avoids code duplication by reusing the _get_db_engine() helper
    engine = _get_db_engine()
    
    # ================================================
    # STEP 2: SAVE DATAFRAME TO DATABASE TABLE
    # ================================================
    # to_sql() is pandas method that converts DataFrame to SQL INSERT or CREATE TABLE
    # SQLAlchemy automatically maps pandas dtypes to PostgreSQL types:
    # - int64 → INTEGER, float64 → REAL, object → VARCHAR, bool → BOOLEAN, etc.
    #
    # Parameters explained:
    # - name: Table name in database (created if not exists with "append"/"replace")
    # - con: SQLAlchemy engine that handles actual SQL execution
    # - if_exists: Action to take if table already exists (see docstring)
    # - index=False: Don't write DataFrame index as a separate column
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False
        )
        
        # ================================================
        # STEP 3: CONFIRMATION & SUMMARY STATISTICS
        # ================================================
        # Print success message with data statistics for verification
        print(f"✓ Data saved to PostgreSQL table: {table_name}")
        # Show row count, column count, and mode used for transparency
        print(f"  Rows: {len(df)} | Columns: {len(df.columns)} | Mode: {if_exists}")
        
    except Exception as e:
        # ================================================
        # ERROR HANDLING
        # ================================================
        # Catch any database or SQL errors and provide meaningful error message
        # Common errors: table name invalid, permission denied, connection lost
        print(f"✗ Error saving data to PostgreSQL table '{table_name}':")
        print(f"  {str(e)}")
        raise  # Re-raise exception so caller can handle if needed




def fetch_trigger_day_data(table_name: str = "ivf_data") -> pd.DataFrame:
    """
    Fetch data from PostgreSQL table into a DataFrame
    """

    # ------------------------------------------------
    # 1. POSTGRESQL CONNECTION PARAMETERS
    # ------------------------------------------------
    username = "postgres"
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
    # 3. FETCH DATA FROM DATABASE
    # ------------------------------------------------
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)

    print(f"Data fetched successfully from table '{table_name}'")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ============================================================
# TESTING & EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    """
    Module testing and example usage.
    
    This section demonstrates how to use the module functions:
    1. Load CSV file into pandas DataFrame
    2. Push DataFrame to PostgreSQL ('ivf_data' table)
    3. Fetch data back from PostgreSQL to verify
    4. Display sample rows to confirm successful round-trip
    """
    
    print("\n" + "="*60)
    print("POSTGRES MODULE - TESTING")
    print("="*60)
    
    # ================================================
    # LOAD DATA FROM CSV FILE
    # ================================================
    print("\n1. Loading CSV file...")
    sample_df = pd.read_csv("data/Trigger_day_prediction.csv")
    print(f"   ✓ Loaded {len(sample_df)} rows from CSV")
    
    # ================================================
    #  UPLOAD DATA TO POSTGRESQL
    # ================================================
    print("\n2. Uploading data to PostgreSQL...")
    push_trigger_day_data(sample_df)
    
    # ================================================
    # FETCH DATA BACK FROM POSTGRESQL
    # ================================================
    print("\n3. Fetching data from PostgreSQL...")
    fetched_df = fetch_trigger_day_data()
    
    # ================================================
    # VERIFY DATA INTEGRITY
    # ================================================
    print("\n4. Verifying data integrity...")
    print("   Sample fetched data (first 5 rows):")
    print(fetched_df.head())
    
    # Compare original and fetched data
    if len(sample_df) == len(fetched_df):
        print(f"\n   ✓ Data integrity verified: {len(fetched_df)} rows match")
    else:
        print(f"   ✗ Data mismatch: Original {len(sample_df)} vs Fetched {len(fetched_df)}")
    
    print("\n" + "="*60 + "\n")


