import sys
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract
from utils import prepare_mnist


# def get_spark():
#     return (
#         SparkSession.builder.appName("Data Pipeline")
#         .config("spark.hadoop.io.nativeio.enabled", "false")
#         .config("spark.hadoop.native.lib", "false")
#         .config("spark.driver.extraJavaOptions", "-Djava.library.path=")
#         .config("spark.executor.extraJavaOptions", "-Djava.library.path=")
#         .getOrCreate()
#     )


def get_spark():
    return (
        SparkSession.builder.appName("Data Pipeline")
        # Disable Hadoop native I/O completely
        .config("spark.hadoop.io.nativeio.enabled", "false")
        .config("spark.hadoop.native.lib", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.library.path=")
        .config("spark.executor.extraJavaOptions", "-Djava.library.path=")
        # Force Spark to use local FS implementations
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.file.impl",
            "org.apache.hadoop.fs.local.LocalFs",
        )
        .getOrCreate()
    )


def run_titanic_pipeline(spark, raw_path, processed_path):
    df = spark.read.csv(raw_path, header=True, inferSchema=True)

    # Handle missing values
    mean_age = df.agg({"Age": "mean"}).collect()[0][0]
    df = df.withColumn("Age", when(col("Age").isNull(), mean_age).otherwise(col("Age")))
    df = df.withColumn(
        "Embarked", when(col("Embarked").isNull(), "S").otherwise(col("Embarked"))
    )
    df = df.fillna({"Fare": 0})

    # Feature engineering
    df = df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\.", 1))
    df = df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    df = df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))

    # df.write.mode("overwrite").parquet(processed_path)
    df.write.mode("overwrite").csv(processed_path, header=True)
    print(f"[Titanic] Processed data saved to {processed_path}")


def run_mnist_pipeline(spark, images_path, labels_path, processed_path):
    df = prepare_mnist(images_path, labels_path, spark)
    # df.write.mode("overwrite").parquet(processed_path)
    df.write.mode("overwrite").csv(processed_path, header=True)
    print(f"[MNIST] Processed data saved to {processed_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--params":
        print("Usage: python src/data_pipeline.py --params params.yaml")
        sys.exit(1)

    params_path = sys.argv[2]
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    spark = get_spark()
    dataset_type = params["dataset"]["type"]

    if dataset_type == "titanic":
        raw_path = params["dataset"]["titanic"]["raw_path"]
        processed_path = params["dataset"]["titanic"]["processed_path"]
        run_titanic_pipeline(spark, raw_path, processed_path)

    elif dataset_type == "mnist":
        images_path = params["dataset"]["mnist"]["raw_images"]
        labels_path = params["dataset"]["mnist"]["raw_labels"]
        processed_path = params["dataset"]["mnist"]["processed_train"]
        run_mnist_pipeline(spark, images_path, labels_path, processed_path)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    spark.stop()
