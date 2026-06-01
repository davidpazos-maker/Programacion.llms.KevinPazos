import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering


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
