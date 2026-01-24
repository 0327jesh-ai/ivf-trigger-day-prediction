# =====================================================
# DISTRIBUTION VALIDATION USING POSTGRES + GREAT EXPECTATIONS
# IVF Trigger Day Prediction
# =====================================================
# Purpose:
# - Validate preprocessed IVF data directly from Postgres
# - Ensure input matches training distribution
# - Protect model from data drift or invalid inputs
# =====================================================

# Core Libraries
import os                                        # Operating system interactions (file/directory operations)
import pandas as pd                             # Data manipulation and DataFrame operations

# Data Validation & Quality
import great_expectations as ge                 # Data quality validation and expectation framework
from great_expectations.exceptions import GreatExpectationsError  # GE exception handling
from great_expectations.core import ExpectationSuite  # Expectation suite data structure

# Database Connectivity
from sqlalchemy import create_engine            # Create database connections and engines
from urllib.parse import quote_plus             # URL-encode special characters in passwords


def get_postgres_engine():
    """
    Create SQLAlchemy engine for PostgreSQL connection
    
    Purpose:
    - Establish database connection for reading data
    - Returns reusable engine for all database operations
    - Enables pandas to query tables directly from PostgreSQL
    
    Returns:
        SQLAlchemy Engine: Database connection object for SQL operations
    """
    
    # PostgreSQL connection credentials
    username = "postgres"                       # Database user
    password = quote_plus("@Gorgeous2703")     # URL-encode password (handles special chars like @)
    host = "localhost"                          # Database server address
    port = "5432"                               # PostgreSQL default port
    database = "ivf"                            # Target database name

    # Create SQLAlchemy engine with connection string
    # Format: postgresql+psycopg2://user:password@host:port/database
    # psycopg2 is the Python adapter for PostgreSQL
    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    )
    
    # Confirm successful connection
    print("✓ Connected to PostgreSQL")
    return engine


def validate_distribution_postgres(table_name: str, mode="inference"):
    """
    Validate preprocessed IVF data distribution using Great Expectations
    
    Purpose:
    - Create data quality expectations from training data (mode="train")
    - Validate inference data against training expectations (mode="inference")
    - Detect data drift, missing values, and out-of-range values
    - Protect model from invalid or unexpected input distributions

    Parameters:
        table_name (str): PostgreSQL table to validate
                         Examples: "ivf_preprocessed_train", "ivf_preprocessed_test"
        mode (str): Operational mode - "train" or "inference"
                   "train": Create expectation suite from training data baseline
                   "inference": Validate new data against saved training expectations
    
    Returns:
        bool: True if validation passes, raises GreatExpectationsError if fails
    """

    # =====================================================
    # STEP 1: CONNECT TO POSTGRESQL DATABASE
    # =====================================================
    # Purpose: Establish connection and load data from database
    
    # Create SQLAlchemy engine for database connectivity
    engine = get_postgres_engine()
    
    # Load entire table into pandas DataFrame
    # pd.read_sql_table() reads table directly from database
    # More efficient than reading to CSV and then loading
    df = pd.read_sql_table(table_name, con=engine)
    print(f"✓ Loaded table '{table_name}' from Postgres with {df.shape[0]} rows")

    # =====================================================
    # STEP 2: CONVERT PANDAS DATAFRAME TO GREAT EXPECTATIONS FORMAT
    # =====================================================
    # Purpose: Create GE DataFrame that supports expectation validation
    
    # Convert pandas DataFrame to Great Expectations DataFrame
    # GE DataFrame wraps pandas but adds validation methods
    # Enables expect_* methods for defining data quality rules
    ge_df = ge.from_pandas(df)

    # Define unique name for expectation suite
    # Suite = collection of data quality rules and thresholds
    # Used to identify and load saved expectations
    suite_name = "ivf_preprocessed_distribution_suite"

    # =====================================================
    # STEP 3: DEFINE EXPECTATIONS FROM TRAINING DATA (TRAIN MODE)
    # =====================================================
    # Purpose: Create baseline data quality rules from training data distribution
    # These expectations define the "acceptable" data range for production data
    # Captures: columns, null values, numeric ranges, cluster assignments
    
    if mode == "train":
        print(f"\n→ Creating expectation suite from training data...")

        # -------------------------------------------------
        # 3A: CREATE EXPECTATION SUITE OBJECT
        # -------------------------------------------------
        # Extract the expectation suite from the GE DataFrame
        # This object holds all expectations/rules we define below
        # discard_failed_expectations=False: Keep all expectations even if they fail
        suite = ge_df.get_expectation_suite(discard_failed_expectations=False)
        
        # Assign a unique name to the suite for identification and reuse
        # This name is used later to load expectations for inference validation
        suite.expectation_suite_name = suite_name
        print(f"  → Expectation suite initialized: {suite_name}")

        # -------------------------------------------------
        # 3B: COLUMN EXISTENCE & NULL VALUE CHECKS
        # -------------------------------------------------
        # Verify all expected columns are present and contain valid data
        # Purpose: Catch missing columns and data quality issues early
        print(f"  → Validating column structure and data completeness...")
        
        for col in df.columns:
            # Expectation 1: Column must exist in dataset
            # Catches cases where columns are dropped or renamed
            ge_df.expect_column_to_exist(col)
            
            # Expectation 2: Column values must not be null
            # Ensures all columns have complete data (no missing values)
            # Production data should have no gaps
            ge_df.expect_column_values_to_not_be_null(col)

        # -------------------------------------------------
        # 3C: NUMERIC FEATURE DISTRIBUTION STATISTICS
        # -------------------------------------------------
        # Capture statistics of numeric features for drift detection
        # Purpose: Detect when feature distributions shift in production
        # This indicates potential data drift or changing patient populations
        print(f"  → Validating numeric feature distributions...")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        
        for col in numeric_cols:
            # -----------------------------------------
            # 3C1: Mean Value Range Check
            # -----------------------------------------
            # Expect column mean to stay within ±20% of training mean
            # Range: 80% to 120% of training mean
            # Detects systematic shifts in average feature values
            # Example: If training mean=100, allow 80-120 in production
            ge_df.expect_column_mean_to_be_between(
                column=col,
                min_value=df[col].mean() * 0.8,      # 80% of training mean
                max_value=df[col].mean() * 1.2       # 120% of training mean
            )
            
            # -----------------------------------------
            # 3C2: Standard Deviation (Variance) Check
            # -----------------------------------------
            # Expect column std dev to stay within 50-150% of training std dev
            # Range: 0.5x to 1.5x of training standard deviation
            # Detects changes in feature variance/spread
            # Example: If training std=10, allow 5-15 in production
            # Catches: More consistent data, more variable data, different populations
            ge_df.expect_column_stdev_to_be_between(
                column=col,
                min_value=df[col].std() * 0.5,        # 50% of training std
                max_value=df[col].std() * 1.5         # 150% of training std
            )

        # -------------------------------------------------
        # 3D: CLUSTER ASSIGNMENT RANGE VALIDATION
        # -------------------------------------------------
        # Patient cluster column is added during preprocessing
        # Verify cluster IDs stay within expected range
        # Purpose: Ensure inference data gets valid cluster assignments
        # No new/unknown clusters should appear in production
        print(f"  → Validating patient cluster assignments...")
        
        if "patient_cluster" in df.columns:
            # Cluster IDs range from 0 to max cluster ID found in training
            # Example: If 3 clusters found (0,1,2), production should only have 0,1,2
            # No values outside this range allowed
            ge_df.expect_column_values_to_be_between(
                column="patient_cluster",
                min_value=0,                           # Cluster IDs start at 0
                max_value=df["patient_cluster"].max()  # Don't allow new cluster IDs
            )
            print(f"    → Cluster ID range: 0 to {df['patient_cluster'].max()} ✓")
        else:
            print(f"    → patient_cluster column not found (skipped)")

        # -------------------------------------------------
        # 3E: SAVE EXPECTATION SUITE TO PERSISTENT STORAGE
        # -------------------------------------------------
        # Persist expectations to Great Expectations DataContext
        # Purpose: Reuse expectations later for validating inference data
        # DataContext is GE's central registry for suites and validations
        print(f"  → Saving expectation suite to storage...")
        
        context = ge.get_context()
        context.save_expectation_suite(expectation_suite=suite)
        
        # Summary of validations created
        print(f"\n✓ Expectation suite '{suite_name}' successfully created and saved")
        print(f"   {len(df.columns)} columns validated")
        print(f"   {len(numeric_cols)} numeric features (mean ± 20%, std 50-150%)")
        print(f"  ✓ Null value checks: ALL columns must have no missing data")
        if "patient_cluster" in df.columns:
            print(f"  Cluster validation: Patient IDs 0-{df['patient_cluster'].max()}")
        print(f"\n→ Training baseline READY for inference validation\n")
        
        return True

    # =====================================================
    # STEP 4: VALIDATE AGAINST TRAINING EXPECTATIONS (INFERENCE MODE)
    # =====================================================
    # Purpose: Check if inference/test data matches training data distribution
    # Detects data drift and protects model from out-of-distribution inputs
    
    else:
        print(f"\n→ Validating against training expectations...")

        # Load and apply saved expectation suite to current data
        # Compares actual data statistics against baseline expectations
        # Checks: null values, mean/stdev ranges, cluster IDs, column existence
        context = ge.get_context()
        suite = context.get_expectation_suite(suite_name)
        validation_result = ge_df.validate(expectation_suite=suite)

        # Check if all expectations passed
        # validation_result["success"] = True if all checks passed, False otherwise
        if not validation_result["success"]:
            # Extract failed expectations for debugging
            failed_checks = validation_result["results"]
            print(f"\n Distribution validation FAILED:")
            for result in failed_checks:
                if not result["success"]:
                    print(f"  → Failed: {result['expectation_config']['expectation_type']}")
                    if "result" in result:
                        print(f"     {result['result']}")
            
            # Raise exception to halt pipeline if validation fails
            # This prevents bad data from reaching the model
            raise GreatExpectationsError(
                f" Distribution validation failed for table '{table_name}'"
            )

        # All validations passed - data matches training distribution
        print(f"✓ Distribution validation PASSED for table '{table_name}'")
        print(f"  → All {len(validation_result['results'])} expectations met")
        return True


# =====================================================
# MAIN EXECUTION
# =====================================================
# Workflow:
# 1. Create expectations from training data (baseline)
# 2. Validate test/inference data against expectations
# 3. Alert if data drift or distribution shift detected

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("DISTRIBUTION VALIDATION PIPELINE")
        print("="*60)
        
        # =====================================================
        # PHASE 1: CREATE EXPECTATION SUITE FROM TRAINING DATA
        # =====================================================
        # Purpose: Establish baseline distribution from training data
        # This defines the "expected" data ranges and distributions
        # Only run this once when training data is finalized
        
        print("\n PHASE 1: Creating expectations from training data...")
        validate_distribution_postgres(
            table_name="ivf_preprocessed_train",
            mode="train"
        )

        # =====================================================
        # PHASE 2: VALIDATE TEST/INFERENCE DATA
        # =====================================================
        # Purpose: Verify test data matches training distribution
        # Detects:
        # - Missing columns or null values
        # - Feature mean/std deviation shifts (data drift)
        # - Out-of-range cluster assignments
        # Protects model from invalid inputs
        
        print("\n PHASE 2: Validating test data against expectations...")
        validate_distribution_postgres(
            table_name="ivf_preprocessed_test",
            mode="inference"
        )

        # =====================================================
        # SUCCESS: ALL VALIDATIONS PASSED
        # =====================================================
        print("\n" + "="*60)
        print("✓ VALIDATION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nStatus: All data quality checks passed")
        print("Action: Safe to proceed with model training/inference\n")

    except GreatExpectationsError as e:
        # Handle Great Expectations validation errors
        print("\n" + "="*60)
        print(f" VALIDATION FAILED")
        print("="*60)
        print(f"Error: {e}")
        print("\nAction: Check data quality before proceeding\n")
        exit(1)

    except Exception as e:
        # Handle unexpected errors (connection issues, missing tables, etc.)
        print("\n" + "="*60)
        print(f" UNEXPECTED ERROR")
        print("="*60)
        print(f"Error: {e}")
        print("\nAction: Check database connection and table existence\n")
        exit(1)
