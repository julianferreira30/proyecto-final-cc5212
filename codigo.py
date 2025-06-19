import matplotlib.pyplot as plt
import pandas as pd

FOLDER = "datos"
FILE = "daily_data_sample.parquet"

df = pd.read_parquet(f"{FOLDER}/{FILE}")
df.head()
df.info()
df.isnull().sum()
df.describe()

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

df_year = (
    df_filtered[["datetime", "temperature_2m_max"]]
    .groupby(["datetime"])
    .mean()
    .reset_index()
)


def estacion(mes_i, mes_f):
    return df_filtered[
        ((df_filtered["month"] >= mes_i) & (df_filtered["month"] <= mes_f))
        | ((df_filtered["month"] == mes_i) & (df_filtered["day"] >= 21))
        | ((df_filtered["month"] == mes_f) & (df_filtered["day"] <= 21))
    ]


mar_jun = estacion(3, 6)
jun_sep = estacion(6, 9)
sep_dic = estacion(9, 12)
dic_mar = estacion(12, 3)

# Dividir por evento meteorologico
mar_jun_tmax = mar_jun[["city_name", "year", "temperature_2m_max"]]
jun_sep_tmax = jun_sep[["city_name", "year", "temperature_2m_max"]]
sep_dic_tmax = sep_dic[["city_name", "year", "temperature_2m_max"]]
dic_mar_tmax = dic_mar[["city_name", "year", "temperature_2m_max"]]

mar_jun_tmin = mar_jun[["city_name", "year", "temperature_2m_min"]]
jun_sep_tmin = jun_sep[["city_name", "year", "temperature_2m_min"]]
sep_dic_tmin = sep_dic[["city_name", "year", "temperature_2m_min"]]
dic_mar_tmin = dic_mar[["city_name", "year", "temperature_2m_min"]]

mar_jun_preci = mar_jun[["city_name", "year", "precipitation_sum"]]
jun_sep_preci = jun_sep[["city_name", "year", "precipitation_sum"]]
sep_dic_preci = sep_dic[["city_name", "year", "precipitation_sum"]]
dic_mar_preci = dic_mar[["city_name", "year", "precipitation_sum"]]

mar_jun_wind = mar_jun[["city_name", "year", "wind_speed_10m_max"]]
jun_sep_wind = jun_sep[["city_name", "year", "wind_speed_10m_max"]]
sep_dic_wind = sep_dic[["city_name", "year", "wind_speed_10m_max"]]
dic_mar_wind = dic_mar[["city_name", "year", "wind_speed_10m_max"]]


# Agrupar por ciudad y calcular los quintiles
def quintiles(df, col):
    return df.groupby("city_name")[col].quantile([0.2, 0.4, 0.6, 0.8]).unstack()
