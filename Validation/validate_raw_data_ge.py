# =====================================================
# DATA VALIDATION USING GREAT EXPECTATIONS
# IVF Trigger Day Prediction
# =====================================================
# Purpose: Validate IVF trigger day dataset quality before processing
# Framework: Great Expectations for automated data quality checks
# Author: ML Trigger Team
# =====================================================

import os
import great_expectations as ge
from great_expectations.exceptions import GreatExpectationsError


def validate_trigger_day_data_ge():
    """
    Validates IVF Trigger Day dataset using Great Expectations framework.
    
    This function performs comprehensive data quality validation including:
    - Table structure and row count checks
    - Column existence and null value verification
    - Numeric range validations (Age, AMH, Day)
    - Categorical value constraints (Trigger_Recommended)
    
    Returns:
        bool: True if all validations pass
        
    Raises:
        GreatExpectationsError: If any validation fails, stops the pipeline
    """

    # ================================================
    # DATA PATH INITIALIZATION
    # ================================================
    # Resolve the absolute path to the CSV file
    # __file__ = current script location
    # dirname() twice = go up from Validation/ to project root
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Construct full path: /project_root/data/Trigger_day_prediction.csv
    data_path = os.path.join(
        project_root,
        "data",
        "Trigger_day_prediction.csv"
    )

    # ================================================
    # LOAD DATA INTO GREAT EXPECTATIONS
    # ================================================
    # Load CSV file into Great Expectations DataFrame
    # This special dataframe includes validation methods
    df = ge.read_csv(data_path)

    print("✓ Dataset loaded into Great Expectations")

    # ================================================
    # DEFINE DATA QUALITY EXPECTATIONS
    # ================================================
    
    # --- Table-Level Validations ---
    # Ensure dataset contains at least 1 row (not empty)
    # Minimum value of 1 ensures we have data to work with
    df.expect_table_row_count_to_be_between(min_value=1)

    # --- Patient Identifier Column ---
    # Verify Patient_ID column exists in the dataset schema
    df.expect_column_to_exist("Patient_ID")
    
    # Ensure every patient has a non-null identifier
    # Null Patient_IDs would make data traceability impossible
    df.expect_column_values_to_not_be_null("Patient_ID")

    # --- Age Column Validation ---
    # Age must fall within realistic IVF procedure range
    # Min: 18 years (legal/biological minimum for reproduction)
    # Max: 50 years (typical fertility treatment age limit)
    # Values outside this range indicate data entry errors
    df.expect_column_values_to_be_between(
        column="Age",
        min_value=18,
        max_value=50
    )

    # --- AMH (Anti-Müllerian Hormone) Column ---
    # AMH is a key biomarker measured in ng/mL
    # Indicates ovarian reserve/egg quality
    # Must be non-negative (zero or positive values only)
    # Negative values are physically impossible and indicate data errors
    df.expect_column_values_to_be_between(
        column="AMH (ng/mL)",
        min_value=0
    )

    # --- Day of Cycle Column ---
    # Represents current day in the IVF monitoring cycle
    # Valid range: Day 1 to Day 20 of cycle monitoring
    # Values outside this range indicate invalid cycle tracking
    df.expect_column_values_to_be_between(
        column="Day",
        min_value=1,
        max_value=20
    )

    # --- Trigger Recommendation Column ---
    # Binary classification column for the target variable
    # 0 = Do NOT trigger HCG injection (not ready)
    # 1 = DO trigger HCG injection (ready for final maturation)
    # Only these two values are valid; any other value is invalid
    df.expect_column_values_to_be_in_set(
        column="Trigger_Recommended (0/1)",
        value_set=[0, 1]
    )

    # ================================================
    # SECTION 4: EXECUTE VALIDATION AND PROCESS RESULTS
    # ================================================
    
    # Run all defined expectations against the dataset
    # Returns a dictionary containing validation results
    validation_result = df.validate()

    # Check if all expectations passed (success flag)
    if not validation_result["success"]:
        # If any validation failed, raise exception to halt pipeline
        # Prevents bad data from flowing to downstream processes
        raise GreatExpectationsError(
            " Data validation failed – stopping pipeline"
        )

    # All validations passed successfully
    # Safe to proceed with data processing
    print(" Data validation passed successfully")

    return True


# ================================================
# MAIN EXECUTION BLOCK
# ================================================

if __name__ == "__main__":
    """
    Entry point for running the validation script directly.
    Executes the full validation workflow for IVF trigger day dataset.
    """
    try:
        # Run the main validation function
        validate_trigger_day_data_ge()
        print("\n✓ Validation pipeline completed successfully")
    except GreatExpectationsError as e:
        # Handle validation failures gracefully
        print(f"\n Validation error: {e}")
        exit(1)
    except Exception as e:
        # Catch unexpected errors
        print(f"\n Unexpected error: {e}")
        exit(1)