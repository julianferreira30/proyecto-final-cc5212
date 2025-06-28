#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Código equivalente a codigo2.py, pero usando PySpark.
Para lanzar en el clúster:
    spark-submit \
      --master yarn \
      --deploy-mode cluster \
      --driver-memory 4G \
      --executor-memory 4G \
      --py-files py_libs.zip \
      codigo2_pyspark.py \
      --parquet hdfs:///datos/daily_data_sample.parquet \
      --cities  hdfs:///datos/cities.csv \
      --out     hdfs:///resultados/resumen_anomalias
"""

import argparse
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    approx_count_distinct,
    col,
    concat_ws,
    dayofmonth,
    expr,
    first,
    lit,
    month,
)
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import when, year

# ────────────────────────────────────────────────
# 0. Parámetros de entrada
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--parquet", required=True, help="Ruta parquet en HDFS")
parser.add_argument("--cities", required=True, help="Ruta cities.csv en HDFS")
parser.add_argument("--out", required=True, help="Carpeta salida CSV")
args = parser.parse_args()

# ────────────────────────────────────────────────
# 1. SparkSession
# ────────────────────────────────────────────────
spark = (
    SparkSession.builder.appName("App")
    .config(
        "spark.sql.parquet.int96RebaseModeInRead", "CORRECTED"
    )  # Forzar a que lea el timestamp como TIMESTAMP_MICROS
    .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .getOrCreate()
)
# ────────────────────────────────────────────────
# 2. Carga de datos
# ────────────────────────────────────────────────
df = spark.read.parquet(args.parquet)
df_cities = (
    spark.read.option("header", "true").option("inferSchema", "true").csv(args.cities)
)

# ────────────────────────────────────────────────
# 3. Pre‑procesamiento básico
# ────────────────────────────────────────────────
cols_keep = [
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
df_filtered = df.select(*cols_keep)

# Convertir a timestamp y desglosar fecha
df_filtered = (
    df_filtered.withColumn("datetime", expr("to_timestamp(datetime)"))
    .withColumn("year", year("datetime"))
    .withColumn("month", month("datetime"))
    .withColumn("day", dayofmonth("datetime"))
)

# Latitud, longitud, hemisferio
df_hemisphere = df_filtered.join(df_cities, "city_name", "left").withColumn(
    "hemisphere", when(col("latitude") > 0, lit("north")).otherwise(lit("south"))
)

# ────────────────────────────────────────────────
# 4. Asignar estación (UDF‑free con when/otherwise)
# ────────────────────────────────────────────────
df_hemisphere = df_hemisphere.withColumn(
    "season",
    when(
        ((col("month") == 3) & (col("day") >= 21))
        | ((col("month") > 3) & (col("month") < 6))
        | ((col("month") == 6) & (col("day") <= 21)),
        lit("mar-jun"),
    )
    .when(
        ((col("month") == 6) & (col("day") >= 21))
        | ((col("month") > 6) & (col("month") < 9))
        | ((col("month") == 9) & (col("day") <= 21)),
        lit("jun-sep"),
    )
    .when(
        ((col("month") == 9) & (col("day") >= 21))
        | ((col("month") > 9) & (col("month") < 12))
        | ((col("month") == 12) & (col("day") <= 21)),
        lit("sep-dic"),
    )
    .otherwise(lit("dic-mar")),
)

# ────────────────────────────────────────────────
# 5. Detección de anomalías con IQR global
# ────────────────────────────────────────────────
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

anomaly_dfs = []
for var in variables:
    # Cuartiles usando approxQuantile (rápido y distribuido)
    q1, q3 = df_hemisphere.approxQuantile(var, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    anomalies = (
        df_hemisphere.filter((col(var) < lower) | (col(var) > upper))
        .withColumn("variable", lit(var))
        .withColumn("measure_unit", lit(measure_units.get(var, "unknown")))
    )
    anomaly_dfs.append(anomalies)

# Unir todas las anomalías
anomaly_df = reduce(lambda x, y: x.unionByName(y), anomaly_dfs)

# ────────────────────────────────────────────────
# 6. Tabla resumen (pivot)
# ────────────────────────────────────────────────
summary_long = (
    anomaly_df.groupBy("city_name", "season", "variable")
    .count()
    .withColumn("season_variable", concat_ws("_", col("season"), col("variable")))
)

summary_pivot = (
    summary_long.groupBy("city_name")
    .pivot("season_variable")
    .agg(first("count"))
    .na.fill(0)
)

summary = summary_pivot.withColumn(
    "total_anomalies",
    reduce(
        lambda a, b: a + b, (col(c) for c in summary_pivot.columns if c != "city_name")
    ),
)

# ────────────────────────────────────────────────
# 7. Guardar CSV
# ────────────────────────────────────────────────
summary.coalesce(1).write.mode("overwrite").option("header", "true").csv(args.out)

spark.stop()
