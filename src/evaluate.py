import os
import sys
import json
import yaml
import argparse

import mlflow
import mlflow.spark

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True, help="Path to validation parquet")
    p.add_argument("--model", required=True, help="Path to saved Spark model dir")
    p.add_argument("--params", required=True, help="Path to params.yaml")
    return p.parse_args()


def get_spark():
    return SparkSession.builder.appName("EvaluationPipeline").getOrCreate()


def save_confusion_matrix_png(pdf: pd.DataFrame, label_col: str, out_path: str):
    # pdf contains 'prediction' and label_col
    cm = pd.crosstab(pdf[label_col], pdf["prediction"], rownames=["Actual"], colnames=["Predicted"], dropna=False)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # annotate
    for (i, j), v in cm.stack().items():
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_titanic(spark, val_path, model_dir, params):
    features_cfg = params["features"]["titanic"]
    label_col = features_cfg["label"]

    val_df = spark.read.parquet(val_path)
    model = PipelineModel.load(model_dir)

    preds = model.transform(val_df)

    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    auc = float(evaluator.evaluate(preds))

    # to pandas for confusion matrix
    pdf = preds.select(col("prediction").cast("double").alias("prediction"),
                       col(label_col).cast("double").alias(label_col)).toPandas()

    # Save metrics.json (DVC)
    metrics = {"AUC": auc}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix artifact
    cm_path = "confusion_matrix.png"
    save_confusion_matrix_png(pdf, label_col, cm_path)

    # MLflow logging
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Titanic"))
    with mlflow.start_run(run_name="evaluate_titanic"):
        mlflow.log_metric("val_auc", auc)
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact(cm_path)

    print(f"[EVAL] Titanic val AUC: {auc:.4f}")
    return auc


def evaluate_mnist(spark, val_path, model_dir, params):
    label_col = params["features"]["mnist"]["label"]
    val_df = spark.read.parquet(val_path)
    model = PipelineModel.load(model_dir)
    preds = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    acc = float(evaluator.evaluate(preds))

    metrics = {"accuracy": acc}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "MNIST"))
    with mlflow.start_run(run_name="evaluate_mnist"):
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_artifact("metrics.json")

    print(f"[EVAL] MNIST val accuracy: {acc:.4f}")
    return acc


def main():
    args = parse_args()
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)

    spark = get_spark()
    ds_type = params["dataset"]["type"]

    if ds_type == "titanic":
        evaluate_titanic(spark, args.val, args.model, params)
    elif ds_type == "mnist":
        evaluate_mnist(spark, args.val, args.model, params)
    else:
        raise ValueError(f"Unsupported dataset: {ds_type}")

    spark.stop()


if __name__ == "__main__":
    main()
