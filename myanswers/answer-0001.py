import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering


def agrupar_clientes_espectralmente(df, n_clusters):
    # 1. Tomar solo las columnas numéricas del DataFrame
    X_num = df.select_dtypes(include=[np.number])

    # 2. Escalar los datos
    X_scaled = StandardScaler().fit_transform(X_num)

    # 3. Aplicar clustering espectral
    modelo = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=3,
        assign_labels="kmeans",
        random_state=42
    )

    # 4. Obtener las etiquetas de cada fila
    labels = modelo.fit_predict(X_scaled)

    # 5. Retornar únicamente un arreglo numpy
    return np.array(labels)
