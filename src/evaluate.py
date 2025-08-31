import argparse
import yaml
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def evaluate_titanic(spark, params):
    processed_path = params["dataset"]["titanic"]["processed_path"]
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
    processed_path = params["dataset"]["mnist"]["processed_train"]
    label_col = params["features"]["mnist"]["label"]

    df = spark.read.parquet(processed_path)
    model = PipelineModel.load("models/mnist_model")

    predictions = model.transform(df)
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    acc = evaluator.evaluate(predictions)

    print(f"MNIST Accuracy = {acc:.4f}")
    return {"accuracy": acc}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=str, required=False, help="Validation parquet path (Titanic)")
    parser.add_argument("--model", type=str, required=False, help="Model path")
    parser.add_argument("--params", type=str, required=True, help="Path to params.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    params_path = args.params

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    spark = SparkSession.builder.appName("EvaluationPipeline").getOrCreate()
    dataset_type = params["dataset"]["type"]

    if dataset_type == "titanic":
        metrics = evaluate_titanic(spark, params)
    elif dataset_type == "mnist":
        metrics = evaluate_mnist(spark, params)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    spark.stop()


if __name__ == "__main__":
    main()
