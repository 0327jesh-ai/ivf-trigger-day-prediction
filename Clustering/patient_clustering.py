# ==================================================
# Unsupervised Clustering – IVF Patient Segmentation
# ==================================================
# Purpose:
# - Find hidden patient groups after preprocessing
# - Select optimal number of clusters using Grid Search
# - Add cluster labels as new ML feature
# ==================================================

# =====================================================
# IMPORTS: Required libraries for clustering and I/O
# =====================================================
import pandas as pd                          # Data manipulation and DataFrame operations
import numpy as np                           # Numerical computing and array operations
from sklearn.cluster import KMeans           # K-Means clustering algorithm implementation
from sklearn.metrics import silhouette_score # Evaluates clustering quality/separation metric
import joblib                                # Model serialization (save/load trained models)
import os                                    # Operating system interactions (file/directory operations)


# =====================================================
# FUNCTION: apply_clustering
# =====================================================
def apply_clustering(
    X: pd.DataFrame,
    k_range=range(2, 9),
    save_model=True
) -> pd.DataFrame:
    """
    Perform KMeans clustering with grid search over K
    and append cluster labels to dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Preprocessed numeric feature matrix
    k_range : range
        Range of cluster values to evaluate (default: 2-8)
    save_model : bool
        Whether to save trained clustering model (default: True)

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional 'patient_cluster' column
        containing cluster assignments for each patient
    """

    # Display process header for logging/monitoring
    print("\n" + "="*60)
    print("CLUSTERING: GRID SEARCH FOR OPTIMAL K")
    print("="*60)

    # =====================================================
    # STEP 1: INITIALIZE VARIABLES FOR GRID SEARCH
    # =====================================================
    # Track the best clustering configuration found
    best_k = None                   # Optimal number of clusters
    best_score = -1                 # Highest silhouette score
    best_model = None               # Trained KMeans model with best performance

    # =====================================================
    # STEP 2: GRID SEARCH USING SILHOUETTE METRIC
    # =====================================================
    # Iterate through all cluster values to find optimal K
    for k in k_range:
        # Initialize KMeans clustering with specified number of clusters
        kmeans = KMeans(
            n_clusters=k,           # Number of clusters for this iteration
            random_state=42,        # Fixed seed for reproducibility
            n_init=10               # Number of times algorithm runs with different centroids
        )
        
        # Fit KMeans model to data and obtain cluster labels
        labels = kmeans.fit_predict(X)

        # Calculate silhouette score (measure of cluster quality: -1 to 1)
        # Higher score = better separated, more cohesive clusters
        score = silhouette_score(X, labels)
        print(f"K={k} → Silhouette Score: {score:.4f}")

        # Update best configuration if current score is better
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans

    # Display results of grid search
    print(f"\n✓ Best K selected: {best_k}")
    print(f"✓ Best Silhouette Score: {best_score:.4f}")

    # =====================================================
    # STEP 3: APPLY BEST MODEL TO GENERATE PREDICTIONS
    # =====================================================
    # Use the optimal model to predict cluster assignments for all patients
    cluster_labels = best_model.predict(X)

    # Create new DataFrame with original features + cluster assignments
    X_clustered = X.copy()
    X_clustered["patient_cluster"] = cluster_labels

    # =====================================================
    # STEP 4: SAVE TRAINED CLUSTERING MODEL FOR FUTURE USE
    # =====================================================
    # Persist model to disk for inference on new data
    if save_model:
        # Create directory for models if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Serialize and save KMeans model using joblib
        joblib.dump(best_model, "models/patient_kmeans.pkl")
        print("✓ Clustering model saved: models/patient_kmeans.pkl")

    # =====================================================
    # RETURN: Clustered dataset with patient group labels
    # =====================================================
    return X_clustered
