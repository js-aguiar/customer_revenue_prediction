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
    #parser.add_argument("--s3_bucket", type=str, help="s3 bucket")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 bucket")
    parser.add_argument("--s3_input_key", type=str, help="s3 input path")
    
    parser.add_argument("--s3_output_bucket", type=str, help="s3 bucket")
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
    input_path = "s3a://" + os.path.join(args.s3_input_bucket, args.s3_input_key)
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
        expr("sum(purchaseamount)").alias("sum_amount"),
        expr("stddev(purchaseamount)").alias("std_amount"),
        expr("last(chain)").alias("chain"),
        countDistinct("dept").alias("num_unique_dept"),
        countDistinct("brand").alias("num_unique_brand"),
    )
    features_all = features_all.withColumn("avg_daily_amount",
                                           expr("sum_amount / (days_since_first_transaction + 1)"))
    features_all = features_all.withColumn("unique_dates_to_days",
                                           expr("num_unique_date / (days_since_first_transaction + 1)"))
    
    
    # Aggregations using time windows (last 180, 90, 60 and 30 days)
    window_features = dict()
    for past_days in [180, 90, 60, 30]:
        key = "{}d".format(past_days)
        window_features[key] = df.filter(df.past_days <= past_days).groupby("id").agg(
            expr("sum(purchaseamount)").alias("purchaseamount_sum_{}d".format(past_days)),
            expr("sum(purchasequantity)").alias("purchasequantity_sum_{}d".format(past_days)),
            expr("count(date_dt)").alias("transactions_count_{}d".format(past_days)),
            expr("avg(purchaseamount)").alias("avg_transaction_amount_{}d".format(past_days)),
            expr("sum(aux_weighted)").alias("time_weighted_amount_{}d".format(past_days)),
            countDistinct("date_dt").alias("unique_dates_{}d".format(past_days)).cast(IntegerType()),
        )
        
    # Aggregations by customer and date
    date_amounts = df.groupby(["id", "date_dt"]).agg(expr("sum(purchaseamount)").alias("daily_amount"))
    date_features = date_amounts.groupby("id").agg(
        expr("avg(daily_amount)").alias("avg_date_amount"),
        expr("stddev(daily_amount)").alias("std_date_amount"),
    )
    
    # Join all features
    features = features_all.join(date_features, on="id", how="left")
    features = features.join(window_features["180d"], on="id", how="left")
    features = features.join(window_features["90d"], on="id", how="left")
    features = features.join(window_features["60d"], on="id", how="left")
    features = features.join(window_features["30d"], on="id", how="left")
    features = features.fillna(0)
    
    # Ratio between features - note: we add a constant to avoid zero division
    features = features.withColumn("recency", expr("days_since_first_transaction - days_since_last_transaction"))
    features = features.withColumn("dslt_to_recency", expr("days_since_last_transaction / (recency + 1)"))
    features = features.withColumn("dslt_to_dsft", expr("days_since_first_transaction / (days_since_first_transaction + 1)"))
    features = features.withColumn("recency_to_unique_dates", expr("recency / num_unique_date"))
    features = features.withColumn("dslt_to_rtud", expr("days_since_last_transaction / (recency_to_unique_dates + 1)"))
    
    features = features.withColumn("sum_amount_to_dsft", expr("sum_amount / (days_since_first_transaction + 1)"))
    features = features.withColumn("sum_amount_to_recency", expr("sum_amount / (recency + 1)"))
    
    features = features.withColumn("amount_30_to_180_rate", expr("purchase_amount_sum_30d / (purchase_amount_sum_180d + 1)"))
    features = features.withColumn("amount_60_to_180_rate", expr("purchase_amount_sum_60d / (purchase_amount_sum_180d + 1)"))
    features = features.withColumn("amount_90_to_180_rate", expr("purchase_amount_sum_90d / (purchase_amount_sum_180d + 1)"))
    features = features.withColumn("amount_30_to_90_rate", expr( "purchase_amount_sum_30d / (purchase_amount_sum_90d + 1)"))

    # Save features as parquet file in S3
    output_path = "s3a://" + os.path.join(args.s3_output_bucket, args.s3_output_key)
    features.repartition(1).write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()