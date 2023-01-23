# Databricks notebook source
comments = spark.read.parquet('dbfs:/FileStore/comments_merged.parquet')

# COMMAND ----------

comments_new_column = comments.withColumn("dollar_sign", comments["body"].rlike("\$"))
comments_new_column = comments_new_column.withColumn("ellipsis", comments_new_column["body"].rlike("(\.{3})"))

# COMMAND ----------

labor_words = ["wage", "salary", "employ", "unemploy", "job", "labor", "labour"]
regex_labor = "(?i)\\b"+"|\\b".join(labor_words)
comments_new_column = comments_new_column.withColumn("labor", comments_new_column["body"].rlike(regex_labor))

# COMMAND ----------

regex_labor

# COMMAND ----------

yolo_words = ["yolo", "all in", "life saving", "leverage"]
regex_yolo = "(?i)\\b"+"|\\b".join(yolo_words)
comments_new_column = comments_new_column.withColumn("yolo", comments_new_column["body"].rlike(regex_yolo))

# COMMAND ----------

btd_words = ["btd", "btfd"]
regex_btd = "(?i)\\b"+"|\\b".join(btd_words)
comments_new_column = comments_new_column.withColumn("btd", comments_new_column["body"].rlike(regex_btd))

# COMMAND ----------

comments_new_column.show()

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import avg, col
comments_new_column.groupBy().agg(avg(col("dollar_sign").cast("int")).alias("dollar"),
                                  avg(col("ellipsis").cast("int")).alias("ellipsis"),
                                  avg(col("labor").cast("int")).alias("labor"),
                                  avg(col("yolo").cast("int")).alias("yolo"),
                                  F.sum(col("btd").cast("int")).alias("btd_sum")).show()
