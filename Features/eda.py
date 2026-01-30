# =============================
#  EXPLORATORY DATA ANALYSIS (EDA)
# =============================

def run_eda_ivf(
    df,
    target_col="Trigger_Recommended (0/1)",
    output_dir="eda_reports",
    save_reports=True,
    show_plots=True
):
    """
    Perform Exploratory Data Analysis for IVF Trigger Prediction
    """

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("EDA: DATA OVERVIEW")
    print("=" * 60)

    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)

    print("\nSample Records:")
    print(df.head())

    # =============================
    # Missing Values Analysis
    # =============================
    print("\n" + "=" * 60)
    print("EDA: MISSING VALUES")
    print("=" * 60)

    missing_df = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (df.isnull().sum() / len(df)) * 100
    }).sort_values(by="missing_percentage", ascending=False)

    print(missing_df[missing_df["missing_count"] > 0])

    # =============================
    # Duplicate Records
    # =============================
    print("\n" + "=" * 60)
    print("EDA: DUPLICATES")
    print("=" * 60)

    print(f"Duplicate rows: {df.duplicated().sum()}")

    # =============================
    # Column Classification
    # =============================
    
    # Select only continuous numerical features for outlier analysis
    # - Exclude target column to prevent data leakage
    # - Exclude binary / low-cardinality columns (e.g., 0/1 flags)
    # - IQR-based outlier detection is meaningful only for continuous variables
    num_cols = [
    col for col in df.select_dtypes(include=np.number).columns
    if df[col].nunique() > 2 and col != target_col
]
    # Select categorical features
    # - Includes object and category dtypes
    # - Useful for value counts and distribution analysis
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # =============================
    # Numerical Summary
    # =============================
    print("\n" + "=" * 60)
    print("EDA: NUMERICAL SUMMARY")
    print("=" * 60)

    numeric_summary = df[num_cols].describe().T
    print(numeric_summary)

    # =============================
    # Categorical Summary
    # =============================
    print("\n" + "=" * 60)
    print("EDA: CATEGORICAL SUMMARY")
    print("=" * 60)

    for col in cat_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts())

    # =============================
    # Outlier Detection (IQR)
    # =============================
    print("\n" + "=" * 60)
    print("EDA: OUTLIER ANALYSIS (IQR)")
    print("=" * 60)

    outlier_info = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outlier_info[col] = outlier_count

        print(f"{col}: {outlier_count} outliers")

    outlier_df = pd.DataFrame.from_dict(
        outlier_info, orient="index", columns=["outlier_count"]
    )

    # =============================
    # Visual Analysis
    # =============================
    if show_plots:
        for col in num_cols:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

        # Correlation heatmap
        if len(num_cols) > 1:
            plt.figure(figsize=(10, 6))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            plt.show()

        # Target relationship
        if target_col in df.columns:
            for col in num_cols:
                if col != target_col:
                    plt.figure()
                    sns.boxplot(x=df[target_col], y=df[col])
                    plt.title(f"{col} vs {target_col}")
                    plt.show()

    # =============================
    # Save EDA Artifacts
    # =============================
    if save_reports:
        numeric_summary.to_csv(f"{output_dir}/numeric_summary.csv")
        missing_df.to_csv(f"{output_dir}/missing_summary.csv")
        outlier_df.to_csv(f"{output_dir}/outlier_summary.csv")

    print("\n✓ EDA COMPLETED SUCCESSFULLY")

    return {
        "numeric_summary": numeric_summary,
        "missing_summary": missing_df,
        "outlier_summary": outlier_df
    }
