from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
df = spark.createDataFrame([(1,'a'), (2,'b')], ["id","val"])
df.show()
