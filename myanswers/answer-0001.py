import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def agrupar_clientes_espectralmente(df, n_clusters):
    X_num = df.select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X_num)

    modelo = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=3,
        assign_labels="kmeans",
        random_state=42
    )

    labels = modelo.fit_predict(X_scaled)
    return np.array(labels)


def permutation_importance_mean(X: pd.DataFrame, y: np.ndarray, n_repeats: int) -> np.ndarray:
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42
    )

    return result.importances_mean
