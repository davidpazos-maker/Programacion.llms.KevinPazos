import numpy as np
import pandas as pd


def _extraer_episodios_alerta_referencia(df, equipo_col, tiempo_col, alerta_col):
    data = df.copy()
    data[tiempo_col] = pd.to_datetime(data[tiempo_col])
    data = data.sort_values([equipo_col, tiempo_col]).reset_index(drop=True)

    cambio_equipo = data[equipo_col].ne(data[equipo_col].shift())
    cambio_alerta = data[alerta_col].ne(data[alerta_col].shift())
    bloque = (cambio_equipo | cambio_alerta).cumsum()
    data["_bloque"] = bloque

    episodios = data[data[alerta_col] == 1].copy()

    if episodios.empty:
        return pd.DataFrame(columns=[equipo_col, "inicio_alerta", "fin_alerta", "num_muestras"])

    salida = (
        episodios.groupby([equipo_col, "_bloque"])
        .agg(
            inicio_alerta=(tiempo_col, "min"),
            fin_alerta=(tiempo_col, "max"),
            num_muestras=(tiempo_col, "size"),
        )
        .reset_index()
        .drop(columns="_bloque")
        .sort_values([equipo_col, "inicio_alerta"])
        .reset_index(drop=True)
    )
    return salida



def generar_caso_de_uso_extraer_episodios_alerta():
    rng = np.random.default_rng()

    equipos = [f"EQ_{i}" for i in range(1, int(rng.integers(3, 6)))]
    rows = []
    base_time = pd.Timestamp("2026-02-01 08:00:00")

    for equipo in equipos:
        n_points = int(rng.integers(10, 18))
        tiempos = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n_points)]

        alerta = []
        estado = int(rng.integers(0, 2))
        for _ in range(n_points):
            if rng.random() < 0.25:
                estado = 1 - estado
            alerta.append(estado)

        if 1 not in alerta:
            alerta[int(rng.integers(0, n_points))] = 1

        for t, a in zip(tiempos, alerta):
            rows.append({"equipo": equipo, "tiempo": t, "alerta": int(a)})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

    input_data = {
        "df": df.copy(),
        "equipo_col": "equipo",
        "tiempo_col": "tiempo",
        "alerta_col": "alerta",
    }
    output_data = _extraer_episodios_alerta_referencia(**input_data)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_extraer_episodios_alerta()
    print("=== INPUT ===")
    print(entrada["df"].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(salida)
