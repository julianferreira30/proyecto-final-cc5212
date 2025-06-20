import numpy as np
import pandas as pd

# 1. Cargar datos
df = pd.read_parquet("datos/daily_data_sample.parquet")
df["datetime"] = pd.to_datetime(df["datetime"])


# 2. Asignar estación según fecha (independiente del año)
def get_season(date):
    y = 2000  # año arbitrario para comparar solo mes y día
    date_fixed = pd.Timestamp(year=y, month=date.month, day=date.day)
    if pd.Timestamp(y, 3, 21) <= date_fixed < pd.Timestamp(y, 6, 21):
        return "Otoño"
    elif pd.Timestamp(y, 6, 21) <= date_fixed < pd.Timestamp(y, 9, 21):
        return "Invierno"
    elif pd.Timestamp(y, 9, 21) <= date_fixed < pd.Timestamp(y, 12, 21):
        return "Primavera"
    else:
        return "Verano"


df["season"] = df["datetime"].apply(get_season)
df["year"] = df["datetime"].dt.year

# 3. Filtrar solo el último año disponible
ultimo_anio = df["year"].max()
df_last_year = df[df["year"] == ultimo_anio]

# 4. Variables a evaluar
variables = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "precipitation_hours",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

# 5. Calcular quintiles y marcar extremos (último quintil)
extremos = []

for var in variables:
    grouped = df_last_year.groupby(["city_name", "season"])
    q80 = grouped[var].transform(lambda x: x.quantile(0.8))
    is_extreme = df_last_year[var] >= q80
    df_last_year[f"{var}_extreme"] = is_extreme
    extremos.append(f"{var}_extreme")

# 6. Contar eventos extremos por ciudad
df_last_year["extreme_event"] = df_last_year[extremos].any(axis=1)
conteo_extremos = df_last_year.groupby("city_name")["extreme_event"].sum().reset_index()
conteo_extremos = conteo_extremos.sort_values(by="extreme_event", ascending=False)

# Resultado: ciudades con más eventos extremos
print(conteo_extremos.head(10))
