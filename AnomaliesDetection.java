package org.mdp.spark.cli;

import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import static org.apache.spark.sql.functions.*;

public class AnomaliesDetection {

    private final SparkSession spark;
    private final Dataset<Row> df;
    private final Dataset<Row> dfCities;

    public AnomaliesDetection(SparkSession spark, String parquetPath, String citiesPath) {
        this.spark = spark;
        this.df = spark.read().parquet(parquetPath);
        this.dfCities = spark.read().option("header", true).csv(citiesPath);
    }

    public Dataset<Row> run() {
        Dataset<Row> dfFiltered = df.select(
            "city_name", "datetime", "weather_code", "temperature_2m_max",
            "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_sum", "precipitation_hours", "rain_sum", "snowfall_sum",
            "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum"
        ).withColumn("datetime", to_timestamp(col("datetime")))
         .withColumn("year", year(col("datetime")))
         .withColumn("month", month(col("datetime")))
         .withColumn("day", dayofmonth(col("datetime")));

        Dataset<Row> dfHemisphere = dfFiltered.join(dfCities, "city_name")
            .withColumn("latitude", col("latitude").cast("double"))
            .withColumn("hemisphere", when(col("latitude").gt(0), "north").otherwise("south"))
            .withColumn("season", when(
                (col("month").equalTo(3).and(col("day").geq(21)))
                .or(col("month").between(4, 5))
                .or(col("month").equalTo(6).and(col("day").leq(21))), "mar-jun")
                .when((col("month").equalTo(6).and(col("day").geq(22)))
                .or(col("month").between(7, 8))
                .or(col("month").equalTo(9).and(col("day").leq(21))), "jun-sep")
                .when((col("month").equalTo(9).and(col("day").geq(22)))
                .or(col("month").between(10, 11))
                .or(col("month").equalTo(12).and(col("day").leq(21))), "sep-dic")
                .otherwise("dic-mar"));

        String[] variables = {
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max"
        };

        Dataset<Row> allAnomalies = null;

        for (String var : variables) {
            Column lowerQ = callUDF("percentile_approx", col(var), lit(0.25));
            Column upperQ = callUDF("percentile_approx", col(var), lit(0.75));
            Dataset<Row> quartiles = dfHemisphere.agg(
                lowerQ.alias("q1"),
                upperQ.alias("q3")
            );
            Row qRow = quartiles.first();
            double q1 = qRow.getDouble(0);
            double q3 = qRow.getDouble(1);
            double iqr = q3 - q1;
            double lowerBound = q1 - 1.5 * iqr;
            double upperBound = q3 + 1.5 * iqr;

            Dataset<Row> anomalies = dfHemisphere.filter(
                col(var).lt(lowerBound).or(col(var).gt(upperBound))
            ).withColumn("variable", lit(var));

            allAnomalies = allAnomalies == null ? anomalies : allAnomalies.union(anomalies);
        }

        Dataset<Row> allAnomaliesWithSeasonVar = allAnomalies.withColumn(
            "season_variable", concat_ws("_", col("season"), col("variable"))
        );

        Dataset<Row> summary = allAnomaliesWithSeasonVar.groupBy("city_name", "season_variable")
            .count()
            .groupBy("city_name")
            .pivot("season_variable")
            .agg(first("count"))
            .na().fill(0);

        Dataset<Row> result = summary.withColumn("total_anomalies", expr("stack(1, *)").getField("col1").cast("int"));
        result.write().option("header", true).csv("resumen_anomalias_output");

        return result;
    }
}