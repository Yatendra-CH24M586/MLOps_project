# import sys
# import yaml
# import json
# from pyspark.sql import SparkSession
# from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
# from pyspark.ml.feature import VectorAssembler, StringIndexer
# from pyspark.ml import Pipeline
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
# import argparse


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", type=str, required=True, help="Path to training parquet file")
#     parser.add_argument("--val", type=str, required=True, help="Path to validation parquet file")
#     parser.add_argument("--params", type=str, required=True, help="Path to params.yaml")
#     return parser.parse_args()


# def train_titanic(spark, train_path, val_path, params):
#     features_cfg = params["features"]["titanic"]
#     model_cfg = params["model"]["titanic"]

#     df = spark.read.parquet(train_path)

#     # Handle categorical features
#     indexers = [
#         StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
#         for col in features_cfg["categorical"]
#     ]

#     feature_cols = features_cfg["numerical"] + [f"{col}_idx" for col in features_cfg["categorical"]]
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#     rf = RandomForestClassifier(
#         featuresCol="features",
#         labelCol=features_cfg["label"],
#         numTrees=model_cfg["numTrees"],
#         maxDepth=model_cfg["maxDepth"],
#     )

#     pipeline = Pipeline(stages=indexers + [assembler, rf])
#     model = pipeline.fit(df)

#     # Save model
#     model.write().overwrite().save("models/titanic_model")

#     print("Titanic model trained and saved at models/titanic_model")
#     return model, df, features_cfg["label"]


# def train_mnist(spark, train_path, val_path, params):
#     df = spark.read.parquet(train_path)
#     label_col = params["features"]["mnist"]["label"]
#     model_cfg = params["model"]["mnist"]

#     feature_cols = [c for c in df.columns if c != label_col]
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#     lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=model_cfg["maxIter"])
#     pipeline = Pipeline(stages=[assembler, lr])
#     model = pipeline.fit(df)

#     model.write().overwrite().save("models/mnist_model")
#     print("MNIST model trained and saved at models/mnist_model")
#     return model, df, label_col


# def main():
#     args = parse_args()
#     train_path = args.train
#     val_path = args.val
#     params_path = args.params

#     with open(params_path, "r") as f:
#         params = yaml.safe_load(f)

#     dataset_type = params["dataset"]["type"]
#     spark = SparkSession.builder.appName("TrainingPipeline").getOrCreate()

#     if dataset_type == "titanic":
#         model, df, label = train_titanic(spark, train_path, val_path, params)
#         evaluator = BinaryClassificationEvaluator(labelCol=label)
#         auc = evaluator.evaluate(model.transform(df))
#         print(f"Training AUC: {auc:.4f}")
#         metrics = {"AUC": auc}

#     elif dataset_type == "mnist":
#         model, df, label = train_mnist(spark, train_path, val_path, params)
#         evaluator = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")
#         acc = evaluator.evaluate(model.transform(df))
#         print(f"Training Accuracy: {acc:.4f}")
#         metrics = {"accuracy": acc}

#     else:
#         raise ValueError(f"Unsupported dataset type: {dataset_type}")

#     # Save metrics
#     with open("metrics.json", "w") as f:
#         json.dump(metrics, f, indent=4)

#     spark.stop()


# if __name__ == "__main__":
#     main()

# src/train.py
import os
import sys
import time
import json
import yaml
import argparse

import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to training parquet")
    p.add_argument("--val", required=True, help="Path to validation parquet")
    p.add_argument("--params", required=True, help="Path to params.yaml")
    return p.parse_args()


def get_spark():
    return SparkSession.builder.appName("TrainingPipeline").getOrCreate()


def _extract_rf_importances(pipeline_model, feature_cols):
    """Try to extract feature importances from the RF stage, if present."""
    try:
        from pyspark.ml.classification import RandomForestClassificationModel
        for st in pipeline_model.stages:
            if isinstance(st, RandomForestClassificationModel):
                importances = st.featureImportances
                # map to names
                pairs = sorted(
                    [(feature_cols[i], float(importances[i])) for i in range(len(feature_cols))],
                    key=lambda x: x[1],
                    reverse=True
                )
                return pairs
    except Exception:
        pass
    return []


def train_titanic(spark, params, train_path, val_path):
    features_cfg = params["features"]["titanic"]
    model_cfg = params["model"]["titanic"]
    label_col = features_cfg["label"]

    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)

    # Categorical → StringIndexer
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in features_cfg["categorical"]
    ]

    feature_cols = features_cfg["numerical"] + [f"{c}_idx" for c in features_cfg["categorical"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        numTrees=model_cfg["numTrees"],
        maxDepth=model_cfg["maxDepth"],
        seed=42,
    )

    pipeline = Pipeline(stages=indexers + [assembler, rf])

    t0 = time.perf_counter()
    model = pipeline.fit(train_df)
    train_time_s = time.perf_counter() - t0

    # Evaluate on val
    preds = model.transform(val_df)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    val_auc = float(evaluator.evaluate(preds))

    # Save local (for DVC) AND log to MLflow
    model.write().overwrite().save("models/titanic_model")

    # ----- MLflow logging -----
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Titanic"))

    with mlflow.start_run(run_name="train_titanic"):
        # Params
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("numTrees", model_cfg["numTrees"])
        mlflow.log_param("maxDepth", model_cfg["maxDepth"])
        mlflow.log_param("label_col", label_col)
        mlflow.log_param("features_num", ",".join(features_cfg["numerical"]))
        mlflow.log_param("features_cat", ",".join(features_cfg["categorical"]))

        # Metrics
        mlflow.log_metric("train_time_s", train_time_s)
        mlflow.log_metric("val_auc", val_auc)

        # Feature importances
        importances = _extract_rf_importances(model, feature_cols)
        if importances:
            # Save as JSON artifact
            with open("feature_importances.json", "w") as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact("feature_importances.json")

        # Log Spark model
        mlflow.spark.log_model(model, artifact_path="model")

    print(f"[TRAIN] Titanic val AUC: {val_auc:.4f} | train_time_s={train_time_s:.2f}")
    return val_auc


def train_mnist(spark, params, train_path, val_path):
    # Placeholder for MNIST when you enable it later
    model_cfg = params["model"]["mnist"]
    label_col = params["features"]["mnist"]["label"]

    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)

    feature_cols = [c for c in train_df.columns if c != label_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=model_cfg["maxIter"])
    pipeline = Pipeline(stages=[assembler, lr])
    model = pipeline.fit(train_df)
    preds = model.transform(val_df)
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    val_acc = float(evaluator.evaluate(preds))

    model.write().overwrite().save("models/mnist_model")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "MNIST"))
    with mlflow.start_run(run_name="train_mnist"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("maxIter", model_cfg["maxIter"])
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.spark.log_model(model, artifact_path="model")

    print(f"[TRAIN] MNIST val accuracy: {val_acc:.4f}")
    return val_acc


def main():
    args = parse_args()
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)

    spark = get_spark()
    ds_type = params["dataset"]["type"]

    if ds_type == "titanic":
        train_titanic(spark, params, args.train, args.val)
    elif ds_type == "mnist":
        train_mnist(spark, params, args.train, args.val)
    else:
        raise ValueError(f"Unsupported dataset: {ds_type}")

    spark.stop()


if __name__ == "__main__":
    main()
