from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


def agrupar_senales_sismicas(df, n_clusters):
    # 1. Eliminar filas con valores nulos
    df_clean = df.dropna()

    # 2. Escalar los datos numéricos
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_clean)

    # 3. Reducir dimensionalidad con Isomap
    isomap = Isomap(n_neighbors=5, n_components=2)
    datos_transformados = isomap.fit_transform(df_scaled)

    # 4. Agrupar con KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(datos_transformados)

    # 5. Calcular índice Davies-Bouldin
    score_db = davies_bouldin_score(datos_transformados, labels)

    # 6. Retornar exactamente en el orden pedido
    return labels, datos_transformados, score_db
