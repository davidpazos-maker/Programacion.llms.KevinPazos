import numpy as np
import pandas as pd


def _calcular_retencion_semanal_referencia(df, user_col, fecha_col):
    data = df.copy()
    data[fecha_col] = pd.to_datetime(data[fecha_col])
    data["semana"] = data[fecha_col].dt.to_period("W-SUN").dt.start_time

    cohortes = data.groupby(user_col)["semana"].min().rename("cohorte")
    data = data.merge(cohortes, on=user_col, how="left")
    data["semana_relativa"] = ((data["semana"] - data["cohorte"]).dt.days // 7).astype(int)

    conteos = (
        data.groupby(["cohorte", "semana_relativa"])[user_col]
        .nunique()
        .rename("usuarios_activos")
        .reset_index()
    )

    tam_cohorte = (
        conteos[conteos["semana_relativa"] == 0]
        .set_index("cohorte")["usuarios_activos"]
        .rename("tam_cohorte")
    )

    conteos = conteos.merge(tam_cohorte, on="cohorte", how="left")
    conteos["retencion"] = conteos["usuarios_activos"] / conteos["tam_cohorte"]

    tabla = conteos.pivot_table(
        index="cohorte",
        columns="semana_relativa",
        values="retencion",
        aggfunc="first",
        fill_value=0.0,
    )

    tabla = tabla.sort_index().sort_index(axis=1)
    tabla.columns.name = None
    return tabla



def generar_caso_de_uso_calcular_retencion_semanal():
    rng = np.random.default_rng()

    n_users = int(rng.integers(6, 14))
    max_weeks = int(rng.integers(3, 7))
    base_monday = pd.Timestamp("2026-01-05")

    rows = []
    for user_id in range(1, n_users + 1):
        cohorte_offset = int(rng.integers(0, 3))
        cohort_week = base_monday + pd.Timedelta(days=7 * cohorte_offset)

        active_weeks = [0]
        for w in range(1, max_weeks):
            if rng.random() < 0.65:
                active_weeks.append(w)

        for week_rel in active_weeks:
            n_events = int(rng.integers(1, 4))
            week_start = cohort_week + pd.Timedelta(days=7 * week_rel)
            for _ in range(n_events):
                day_offset = int(rng.integers(0, 7))
                hour_offset = int(rng.integers(0, 24))
                rows.append(
                    {
                        "usuario": f"u{user_id:02d}",
                        "fecha": week_start + pd.Timedelta(days=day_offset, hours=hour_offset),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

    input_data = {
        "df": df.copy(),
        "user_col": "usuario",
        "fecha_col": "fecha",
    }
    output_data = _calcular_retencion_semanal_referencia(**input_data)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_calcular_retencion_semanal()
    print("=== INPUT ===")
    print(entrada["df"].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(salida)
