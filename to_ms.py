import pandas as pd

# Ruta local o en HDFS si tienes acceso por fsspec
INPUT = "datos/daily_data_sample.parquet"
OUTPUT = "datos/daily_data_sample_ms.parquet"

# Lee el archivo original
df = pd.read_parquet(INPUT)

# Guarda con timestamps en milisegundos
df.to_parquet(OUTPUT, coerce_timestamps="ms")
