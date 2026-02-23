import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def train_clustering_model(X_train_scaled, X_val_scaled):
    """
    Trains KMeans using selected feature subset.
    Uses training data only for fitting.
    Validation data used for silhouette evaluation.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Selected Feature Subset
    cluster_features = [
        "MedInc",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    X_train_cluster = X_train_scaled[cluster_features]
    X_val_cluster = X_val_scaled[cluster_features]

    # Elbow Method (Training Set Only)
    inertia_values = []
    K_range = range(2, 9)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_cluster)
        inertia_values.append(kmeans.inertia_)

    # Plot Elbow
    plt.figure()
    plt.plot(K_range, inertia_values, marker='o')
    plt.title("Elbow Method (Training Set)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("reports/figures/kmeans_elbow.png")
    plt.close()

    # Choose K = 3 (Clean and interpretable)
    optimal_k = 3

    final_kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    final_kmeans.fit(X_train_cluster)

    # Silhouette Score (Validation Set)
    val_labels = final_kmeans.predict(X_val_cluster)
    silhouette = silhouette_score(X_val_cluster, val_labels)

    print(f"Clustering Silhouette Score (Validation): {silhouette:.4f}")

    # PCA for Visualization
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_cluster)

    train_labels = final_kmeans.labels_

    plt.figure()
    scatter = plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=train_labels,
        cmap="viridis",
        alpha=0.6
    )
    plt.title("KMeans Clusters (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig("reports/figures/kmeans_clusters_pca.png")
    plt.close()

    # Save Model + Metadata
    joblib.dump(
        {
            "model": final_kmeans,
            "features": cluster_features,
            "n_clusters": optimal_k
        },
        "models/clustering_model.pkl"
    )

    print("Clustering model saved.")