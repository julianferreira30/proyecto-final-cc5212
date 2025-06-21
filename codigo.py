import matplotlib.pyplot as plt
import pandas as pd

FOLDER = "datos"
FILE = "daily_data_sample.parquet"
CITIES_FILE = "cities.csv"

df = pd.read_parquet(f"{FOLDER}/{FILE}")
df.head()
df.info()
df.isnull().sum()
df.describe()

df_cities = pd.read_csv(f"{FOLDER}/{CITIES_FILE}")

# Filtro de columnas
"""
city_name
datetime
weather_code
termperature_2m_max
termperature_2m_min
apparent_temperature_max
apparent_temperature_min
precipitation_sum
precipitation_hours
rain_sum
snowfall_sum
wind_speed_10m_max
wind_gusts_10m_max
shortwave_radiation_sum
"""

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
df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"])

df_filtered["year"] = df_filtered["datetime"].dt.year
df_filtered["month"] = df_filtered["datetime"].dt.month
df_filtered["day"] = df_filtered["datetime"].dt.day

cities = df_filtered.groupby(["city_name"]).mean().reset_index()

# Dataframe que incluye las latitudes y longitudes de las ciudades
df_hemisphere = df_filtered.merge(df_cities, on="city_name")

# Filtrando el hemisferio en el que están las ciudades
df_hemisphere["hemisphere"] = df_hemisphere["latitude"].apply(lambda x: "north" if x > 0 else "south")

df_year = (
    df_filtered[["datetime", "temperature_2m_max"]]
    .groupby(["datetime"])
    .mean()
    .reset_index()
)


def estacion(mes_i, mes_f, df):
    return df[
        ((df["month"] >= mes_i) & (df["month"] <= mes_f))
        | ((df["month"] == mes_i) & (df["day"] >= 21))
        | ((df["month"] == mes_f) & (df["day"] <= 21))
    ]


mar_jun = estacion(3, 6, df_hemisphere)
jun_sep = estacion(6, 9, df_hemisphere)
sep_dic = estacion(9, 12, df_hemisphere)
dic_mar = estacion(12, 3, df_hemisphere)

# Dividir por evento meteorologico
mar_jun_tmax = mar_jun[["city_name", "year", "temperature_2m_max", "hemisphere"]]
jun_sep_tmax = jun_sep[["city_name", "year", "temperature_2m_max", "hemisphere"]]
sep_dic_tmax = sep_dic[["city_name", "year", "temperature_2m_max", "hemisphere"]]
dic_mar_tmax = dic_mar[["city_name", "year", "temperature_2m_max", "hemisphere"]]

mar_jun_tmin = mar_jun[["city_name", "year", "temperature_2m_min", "hemisphere"]]
jun_sep_tmin = jun_sep[["city_name", "year", "temperature_2m_min", "hemisphere"]]
sep_dic_tmin = sep_dic[["city_name", "year", "temperature_2m_min", "hemisphere"]]
dic_mar_tmin = dic_mar[["city_name", "year", "temperature_2m_min", "hemisphere"]]

mar_jun_preci = mar_jun[["city_name", "year", "precipitation_sum", "hemisphere"]]
jun_sep_preci = jun_sep[["city_name", "year", "precipitation_sum", "hemisphere"]]
sep_dic_preci = sep_dic[["city_name", "year", "precipitation_sum", "hemisphere"]]
dic_mar_preci = dic_mar[["city_name", "year", "precipitation_sum", "hemisphere"]]

mar_jun_wind = mar_jun[["city_name", "year", "wind_speed_10m_max", "hemisphere"]]
jun_sep_wind = jun_sep[["city_name", "year", "wind_speed_10m_max", "hemisphere"]]
sep_dic_wind = sep_dic[["city_name", "year", "wind_speed_10m_max", "hemisphere"]]
dic_mar_wind = dic_mar[["city_name", "year", "wind_speed_10m_max", "hemisphere"]]

month_and_event = [mar_jun_tmax, jun_sep_tmax, sep_dic_tmax, dic_mar_tmax,
                   mar_jun_tmin, jun_sep_tmin, sep_dic_tmin, dic_mar_tmin,
                   mar_jun_preci, jun_sep_preci, sep_dic_preci, dic_mar_preci,
                   mar_jun_wind, jun_sep_wind, sep_dic_wind, dic_mar_wind]

measure_units = {"temperature_2m_max": "°C",
                 "temperature_2m_min": "°C",
                 "precipitation_sum": "mm",
                 "wind_speed_10m_max": "m/s"}

# Agrupar por ciudad y calcular los quintiles
def quintiles(df, col):
    return df.groupby("city_name")[col].quantile([0.2, 0.4, 0.6, 0.8]).unstack()

visualizer = []

for df in month_and_event:
    df_quintiles = quintiles(df, df.columns[2])
    df_quintiles.columns = ["Q1", "Q2", "Q3", "Q4"]
    
    # Si es la minima temperatura, se busca el Q1
    if(df.columns[2] == "temperature_2m_min"):
        for anormalidad in df_quintiles["Q1"]:
            ciudad = df_quintiles.index[df_quintiles["Q1"] == anormalidad].tolist()[0]
            print(f"Anomalía detectada en {ciudad}: {anormalidad} {measure_units[df.columns[2]]} en {df.columns[2]}")
            visualizer.append(f"Anomalía detectada en {ciudad}: {anormalidad} {measure_units[df.columns[2]]} en {df.columns[2]}")
    else:
        for anormalidad in df_quintiles["Q4"]:
            ciudad = df_quintiles.index[df_quintiles["Q4"] == anormalidad].tolist()[0]
            print(f"Anomalía detectada en {ciudad}: {anormalidad} {measure_units[df.columns[2]]} en {df.columns[2]}")

print("\n".join(visualizer))
        
"""
        ciudad = df_quintiles.index[df_quintiles["Q4"] == anormalidad].tolist()[0]
        print(f"Anormalidad en {ciudad} para {df.columns[2]}: {anormalidad}")
        df_anom = df[df["city_name"] == ciudad]
        plt.figure(figsize=(10, 5))
        plt.plot(df_anom["year"], df_anom[df.columns[2]], marker='o', linestyle='-')
        plt.title(f"Anomalía de {df.columns[2]} en {ciudad}")
        plt.xlabel("Año")
        plt.ylabel(df.columns[2])
        plt.grid()
        plt.show()
        """

