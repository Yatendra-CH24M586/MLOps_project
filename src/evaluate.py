import sys
import yaml
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def evaluate_titanic(spark, params):
    processed_path = params["dataset"]["processed_path"]
    features_cfg = params["features"]["titanic"]

    # Load data + trained model
    df = spark.read.parquet(processed_path)
    model = PipelineModel.load("models/titanic_model")

    predictions = model.transform(df)

    evaluator = BinaryClassificationEvaluator(labelCol=features_cfg["label"])
    auc = evaluator.evaluate(predictions)

    print(f"Titanic AUC = {auc:.4f}")
    return {"AUC": auc}


def evaluate_mnist(spark, params):
    processed_path = params["dataset"]["processed_path"]
    label_col = params["features"]["mnist"]["label"]

    df = spark.read.parquet(processed_path)
    model = PipelineModel.load("models/mnist_model")

    predictions = model.transform(df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col, metricName="accuracy"
    )
    acc = evaluator.evaluate(predictions)

    print(f"MNIST Accuracy = {acc:.4f}")
    return {"accuracy": acc}


def main(params_path):
    spark = SparkSession.builder.appName("EvaluationPipeline").getOrCreate()

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    dataset_type = params["dataset"]["type"]

    if dataset_type == "titanic":
        metrics = evaluate_titanic(spark, params)
    elif dataset_type == "mnist":
        metrics = evaluate_mnist(spark, params)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    # Save metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        params_path = sys.argv[1]
    else:
        params_path = "params.yaml"

    main(params_path)
