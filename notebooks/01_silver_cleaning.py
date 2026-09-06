# Databricks notebook source
flights_df = spark.table("workspace.default.flights_raw")
airlines_df = spark.table("workspace.default.airlines_raw")
airports_df = spark.table("workspace.default.airports_raw")

print("Flights:", flights_df.count())
print("Airlines:", airlines_df.count())
print("Airports:", airports_df.count())

# COMMAND ----------

from pyspark.sql.functions import col, when

# Fill null delay values with 0, otherwise calculations will break
flights_clean = flights_df.fillna({"DEPARTURE_DELAY": 0, "ARRIVAL_DELAY": 0, "CANCELLED": 0})

# Keep only the columns we actually need out of the original 31
cols_needed = ["YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
               "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY", "ARRIVAL_DELAY", "CANCELLED"]

flights_clean = flights_clean.select(*cols_needed)

# A flight is considered "delayed" if it arrives 15+ minutes late (standard industry definition)
flights_clean = flights_clean.withColumn("IS_DELAYED", when(col("ARRIVAL_DELAY") >= 15, 1).otherwise(0))

print("Total rows after cleaning:", flights_clean.count())
flights_clean.show(5)

# COMMAND ----------

# Save the cleaned data as a Delta table so it persists beyond this notebook
flights_clean.write.format("delta").mode("overwrite").saveAsTable("workspace.default.flights_silver")

print("Silver table created: flights_silver")

# COMMAND ----------

# Register the silver table as a temporary SQL view so we can query it with SQL
flights_clean.createOrReplaceTempView("flights_temp")


# COMMAND ----------

airline_delay_summary = spark.sql("""
    SELECT AIRLINE,
           COUNT(*) AS total_flights,
           SUM(IS_DELAYED) AS total_delayed,
           ROUND(AVG(ARRIVAL_DELAY), 2) AS avg_arrival_delay
    FROM flights_temp
    GROUP BY AIRLINE
    ORDER BY avg_arrival_delay DESC
""")

airline_delay_summary.show()

# COMMAND ----------

# Find airports where the average delay is worse than the overall average delay
# This uses a subquery — the inner query calculates the overall average first
airport_delay_analysis = spark.sql("""
    SELECT ORIGIN_AIRPORT,
           COUNT(*) AS total_flights,
           ROUND(AVG(DEPARTURE_DELAY), 2) AS avg_departure_delay
    FROM flights_temp
    GROUP BY ORIGIN_AIRPORT
    HAVING AVG(DEPARTURE_DELAY) > (
        SELECT AVG(DEPARTURE_DELAY) FROM flights_temp
    )
    ORDER BY avg_departure_delay DESC
    LIMIT 15
""")

airport_delay_analysis.show()


# COMMAND ----------

# Save both insights as Gold tables so they persist and can be used in a dashboard later
airline_delay_summary.write.format("delta").mode("overwrite").saveAsTable("workspace.default.gold_airline_delays")
airport_delay_analysis.write.format("delta").mode("overwrite").saveAsTable("workspace.default.gold_airport_delays")

print("Gold tables created: gold_airline_delays, gold_airport_delays")

# COMMAND ----------

monthly_delay_trend = spark.sql("""
    SELECT MONTH,
           COUNT(*) AS total_flights,
           SUM(IS_DELAYED) AS total_delayed,
           ROUND(SUM(IS_DELAYED) * 100.0 / COUNT(*), 2) AS delay_percentage
    FROM flights_temp
    GROUP BY MONTH
    ORDER BY MONTH
""")

monthly_delay_trend.show(12)

# Save this as a Gold table too
monthly_delay_trend.write.format("delta").mode("overwrite").saveAsTable("workspace.default.gold_monthly_delays")
print("Gold table created: gold_monthly_delays")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Convert airline (text) into a numeric code, since ML models only understand numbers
airline_indexer = StringIndexer(inputCol="AIRLINE", outputCol="AIRLINE_INDEX")

# Combine all the features we want to use into a single vector column
assembler = VectorAssembler(
    inputCols=["MONTH", "DAY_OF_WEEK", "AIRLINE_INDEX", "DEPARTURE_DELAY"],
    outputCol="features"
)

# The actual model
lr = LogisticRegression(featuresCol="features", labelCol="IS_DELAYED")

# Chain all steps together into one pipeline
pipeline = Pipeline(stages=[airline_indexer, assembler, lr])

# Split data into training (80%) and testing (20%)
train_data, test_data = flights_clean.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

print("Model training done")

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Run the trained model on the test data (data it hasn't seen before)
predictions = model.transform(test_data)

# Check how many predictions were correct
evaluator = MulticlassClassificationEvaluator(
    labelCol="IS_DELAYED",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Look at a few actual predictions
predictions.select("MONTH", "AIRLINE", "DEPARTURE_DELAY", "IS_DELAYED", "prediction").show(10)

# COMMAND ----------

# Quick look at all our gold tables together before visualizing
display(spark.table("workspace.default.gold_airline_delays"))

# COMMAND ----------

display(spark.table("workspace.default.gold_monthly_delays"))

# COMMAND ----------

display(spark.table("workspace.default.gold_airport_delays"))
