import argparse
import csv
import os
import shutil
import sys
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_bucket", type=str, help="s3 bucket")
    parser.add_argument("--s3_input_key", type=str, help="s3 input path")
    parser.add_argument("--s3_output_key", type=str, help="s3 output path")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    # Data schema
    schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("chain", ShortType(), True),
            StructField("dept", ByteType(), True),
            StructField("category", ShortType(), True),
            StructField("company", LongType(), True),
            StructField("brand", IntegerType(), True),
            StructField("date", IntegerType(), True),
            StructField("purchasequantity", IntegerType(), True),
            StructField("purchaseamount", DoubleType(), True)
        ]
    )

    # Downloading the data from S3 into a Dataframe
    input_path = "s3a://" + os.path.join(args.s3_bucket, args.s3_input_key)
    df = spark.read.schema(schema).parquet(input_path)
    
    # Convert to date format
    df = df.withColumn("date_dt", to_date(col("date").cast("string"), "yyyyMMdd"))
    
    max_date = df.select(max("date_dt")).first()   # returns a sql.Row object
    max_date = max_date["max(date_dt)"]            # extract a datetime.date object
    
    # Auxiliary columns
    df = df.withColumn("past_days", datediff(lit(max_date), col("date_dt")))
    df = df.withColumn("aux_weighted", expr("purchaseamount / log(past_days + 2)"))
    
    # First aggregation using the entire history
    features_all = df.groupby("id").agg(
        datediff(lit(max_date), max("date_dt")).alias("days_since_last_transaction"),
        datediff(lit(max_date), min("date_dt")).alias("days_since_first_transaction"),
        countDistinct("date_dt").alias("num_unique_date"),
        expr("max(purchaseamount)").alias("max_transaction_amount"),
        expr("avg(purchaseamount)").alias("avg_transaction_amount"),
        expr("sum(purchaseamount)").alias("sum_amount"), # not used
    )
    features_all = features_all.withColumn("avg_daily_amount",
                                           expr("total_amount / days_since_first_transaction"))
    features_all = features_all.withColumn("unique_dates_to_days",
                                           expr("num_unique_date / days_since_first_transaction"))
    
    # Aggregations using time windows (last 180, 90, 60 and 30 days)
    window_features = dict()

    for past_days in [180, 90, 60, 30]:
        key = "{}d".format(past_days)
        window_features[key] = df.filter(df.past_days <= past_days).groupby("id").agg(
            expr("sum(purchaseamount)").alias("purchaseamount_sum_{}d".format(past_days)),
            expr("sum(purchasequantity)").alias("purchasequantity_sum_{}d".format(past_days)),
            expr("count(date_dt)").alias("transactions_count_{}d".format(past_days)),
            expr("avg(purchaseamount)").alias("avg_transaction_amount_{}d".format(past_days)),
            expr("sum(aux_weighted").alias("time_weighted_amount_{}d".format(past_days)),
            countDistinct("date_dt").alias("unique_dates_{}d".format(past_days)),
        )
    
    # Join all features
    features = features_all.join(window_features["180d"], on="id", how="left")
    features = features.join(window_features["90d"], on="id", how="left")
    features = features.join(window_features["60d"], on="id", how="left")
    features = features.join(window_features["30d"], on="id", how="left")
    features = features.fillna(0)

    # Save features as parquet file in S3
    output_path = "s3a://" + os.path.join(args.s3_bucket, args.s3_output_key)
    features.repartition(1).write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()