import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------
# 1. Carga de datos
# ------------------------------------------
FOLDER = "datos"
FILE = "daily_data_sample.parquet"
CITIES_FILE = "cities.csv"

df = pd.read_parquet(f"{FOLDER}/{FILE}")
df_cities = pd.read_csv(f"{FOLDER}/{CITIES_FILE}")

# ------------------------------------------
# 2. Preprocesamiento
# ------------------------------------------
# Columnas relevantes
df_filtered = df[
    [
        "city_name",
        "datetime",
        "weather_code",
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
        "shortwave_radiation_sum",
    ]
]

# Convertir datetime
df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"])
df_filtered["year"] = df_filtered["datetime"].dt.year
df_filtered["month"] = df_filtered["datetime"].dt.month
df_filtered["day"] = df_filtered["datetime"].dt.day

# Agregar latitudes y longitudes
df_hemisphere = df_filtered.merge(df_cities, on="city_name")
df_hemisphere["hemisphere"] = df_hemisphere["latitude"].apply(
    lambda x: "north" if x > 0 else "south"
)


# ------------------------------------------
# 3. Asignar estación del año
# ------------------------------------------
def asignar_estacion(mes, dia):
    if (mes == 3 and dia >= 21) or (3 < mes < 6) or (mes == 6 and dia <= 21):
        return "mar-jun"
    elif (mes == 6 and dia >= 21) or (6 < mes < 9) or (mes == 9 and dia <= 21):
        return "jun-sep"
    elif (mes == 9 and dia >= 21) or (9 < mes < 12) or (mes == 12 and dia <= 21):
        return "sep-dic"
    else:
        return "dic-mar"


df_hemisphere["season"] = df_hemisphere.apply(
    lambda row: asignar_estacion(row["month"], row["day"]), axis=1
)

# ------------------------------------------
# 4. Detectar anomalías por deciles
# ------------------------------------------
variables = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

measure_units = {
    "temperature_2m_max": "°C",
    "temperature_2m_min": "°C",
    "precipitation_sum": "mm",
    "wind_speed_10m_max": "m/s",
}

anomaly_records = []

for var in variables:
    # Calcular deciles (10% y 90%)
    deciles_df = (
        df_hemisphere.groupby(["city_name", "season"])[var]
        .quantile([0.1, 0.9])
        .unstack()
        .reset_index()
    )
    deciles_df.columns = ["city_name", "season", "D1", "D9"]

    # Unir deciles a los datos originales
    merged = df_hemisphere[["city_name", "season", "year", var]].merge(
        deciles_df, on=["city_name", "season"], how="left"
    )

    # Detectar anomalías
    anomalías_bajas = merged[merged[var] <= merged["D1"]].copy()
    anomalías_bajas["type"] = "low"

    anomalías_altas = merged[merged[var] >= merged["D9"]].copy()
    anomalías_altas["type"] = "high"

    anomalías = pd.concat([anomalías_bajas, anomalías_altas])
    anomalías["variable"] = var

    anomaly_records.append(anomalías)

# Concatenar todas las anomalías
anomaly_df = pd.concat(anomaly_records, ignore_index=True)

# ------------------------------------------
# 5. Tabla resumen
# ------------------------------------------
summary = (
    anomaly_df.groupby(["city_name", "season", "variable"])
    .size()
    .reset_index(name="count")
    .pivot_table(
        index="city_name", columns=["season", "variable"], values="count", fill_value=0
    )
)
# agregar columna con la suma de todas las anomalías por ciudad y estación
summary["total_anomalies"] = summary.sum(axis=1)

# Guardar CSV
summary.to_csv("resumen_anomalias.csv")

# ------------------------------------------
# 6. Visualización (heatmap)
# ------------------------------------------
plt.figure(figsize=(30, 20))
sns.heatmap(summary, cmap="Reds", linewidths=0.5, annot=True)
plt.title("Anomalías meteorológicas por ciudad, estación y variable")
plt.tight_layout()
plt.show()
