import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _estimar_covarianza_ledoitwolf_referencia(df, feature_cols):
    X = df[feature_cols].dropna().to_numpy()
    modelo = LedoitWolf()
    modelo.fit(X)

    return pd.DataFrame(
        modelo.covariance_,
        index=feature_cols,
        columns=feature_cols,
    )



def generar_caso_de_uso_estimar_covarianza_ledoitwolf():
    rng = np.random.default_rng()

    n_rows = int(rng.integers(25, 46))
    n_features = int(rng.integers(3, 6))
    feature_cols = [f"feature_{i}" for i in range(n_features)]

    A = rng.normal(size=(n_features, n_features))
    cov = A @ A.T + np.eye(n_features) * 0.5
    mean = rng.normal(loc=0.0, scale=2.0, size=n_features)
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n_rows)

    df = pd.DataFrame(X, columns=feature_cols)

    mask = rng.random(df.shape) < 0.08
    df = df.mask(mask)

    input_data = {
        "df": df.copy(),
        "feature_cols": feature_cols,
    }
    output_data = _estimar_covarianza_ledoitwolf_referencia(**input_data)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_estimar_covarianza_ledoitwolf()
    print("=== INPUT ===")
    print(entrada["df"].head())
    print("feature_cols:", entrada["feature_cols"])
    print("\n=== OUTPUT ESPERADO ===")
    print("\n=== OUTPUT ESPERADO ===")
    print(salida)
