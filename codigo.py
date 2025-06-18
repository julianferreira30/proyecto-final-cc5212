
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('daily_data_sample.parquet')
df.head()
df.info()
df.isnull().sum()
df.describe()

#Filtro de columnas
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

df_filtered = df[[
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
    "shortwave_radiation_sum"
]]

df_gruped = df_filtered.groupby(['city_name']).mean().reset_index()

df_year = df_filtered[["datetime", "temperature_2m_max"]].groupby(["datetime"]).mean().reset_index()