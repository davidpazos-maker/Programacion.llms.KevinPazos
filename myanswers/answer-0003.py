import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


def segmentar_clientes(df, n_clusters=3):
    # 1. Crear una copia del DataFrame original
    df_out = df.copy()

    # 2. Extraer solo las columnas numéricas
    X = df_out.select_dtypes(include=[np.number])

    # 3. Imputar datos faltantes con constante 0
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_imputed = imputer.fit_transform(X)

    # 4. Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_imputed)

    # 5. Añadir la columna segmento_id
    df_out["segmento_id"] = kmeans.labels_

    # 6. Obtener centroides
    centroides = kmeans.cluster_centers_

    # 7. Retornar DataFrame modificado y centroides
    return df_out, centroides
