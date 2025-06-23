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


def iter_cuartil(df_hemisphere, var):
    """Función para calcular el rango intercuartil (IQR)
    y detectar anomalías

    Args:
        df_hemisphere (pd.DataFrame): DataFrame con los datos meteorológicos.
        var (str): Nombre de la variable a analizar.
    Returns:
        pd.DataFrame: DataFrame con las anomalías detectadas.
    """
    # Calcular cuartiles
    q1 = df_hemisphere[var].quantile(0.25)
    q3 = df_hemisphere[var].quantile(0.75)
    iqr = q3 - q1

    # Definir límites para detectar anomalías
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Detectar anomalías
    anomalies = df_hemisphere[
        (df_hemisphere[var] < lower_bound) | (df_hemisphere[var] > upper_bound)
    ]
    anomalies = anomalies.copy()
    anomalies["variable"] = var
    anomalies["measure_unit"] = measure_units.get(var, "unknown")

    return anomalies


anomaly_records = []

for var in variables:
    # Iterar sobre cada variable y detectar anomalías
    anomalies = iter_cuartil(df_hemisphere, var)
    if not anomalies.empty:
        anomaly_records.append(anomalies)

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
