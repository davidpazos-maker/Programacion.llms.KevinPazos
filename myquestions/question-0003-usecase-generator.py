import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def _estimar_densidad_kde_referencia(df, value_col, bandwidth=0.5, num_points=100):
    valores = df[value_col].dropna().to_numpy().reshape(-1, 1)
    grid = np.linspace(valores.min(), valores.max(), num_points).reshape(-1, 1)

    modelo = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    modelo.fit(valores)

    log_density = modelo.score_samples(grid)
    density = np.exp(log_density)

    resultado = pd.DataFrame({
        value_col: grid.ravel(),
        "density": density,
    })
    return resultado.sort_values(by=value_col).reset_index(drop=True)



def generar_caso_de_uso_estimar_densidad_kde():
    rng = np.random.default_rng()

    n1 = int(rng.integers(25, 41))
    n2 = int(rng.integers(20, 36))
    valores = np.concatenate([
        rng.normal(loc=-1.5, scale=0.6, size=n1),
        rng.normal(loc=2.0, scale=0.9, size=n2),
    ])

    rng.shuffle(valores)
    value_col = "medicion"
    df = pd.DataFrame({value_col: valores})

    mask = rng.random(df.shape[0]) < 0.12
    df.loc[mask, value_col] = np.nan

    bandwidth = float(rng.choice([0.25, 0.4, 0.6, 0.8]))
    num_points = int(rng.integers(60, 121))

    input_data = {
        "df": df.copy(),
        "value_col": value_col,
        "bandwidth": bandwidth,
        "num_points": num_points,
    }
    output_data = _estimar_densidad_kde_referencia(**input_data)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_estimar_densidad_kde()
    print("=== INPUT ===")
    print(entrada["df"].head())
    print("value_col:", entrada["value_col"])
    print("bandwidth:", entrada["bandwidth"])
    print("num_points:", entrada["num_points"])
    print("\n=== OUTPUT ESPERADO ===")
    print(salida.head())
