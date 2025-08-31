import sys
import yaml
import json
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training parquet file")
    parser.add_argument("--val", type=str, required=True, help="Path to validation parquet file")
    parser.add_argument("--params", type=str, required=True, help="Path to params.yaml")
    return parser.parse_args()


def train_titanic(spark, train_path, val_path, params):
    features_cfg = params["features"]["titanic"]
    model_cfg = params["model"]["titanic"]

    df = spark.read.parquet(train_path)

    # Handle categorical features
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in features_cfg["categorical"]
    ]

    feature_cols = features_cfg["numerical"] + [f"{col}_idx" for col in features_cfg["categorical"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=features_cfg["label"],
        numTrees=model_cfg["numTrees"],
        maxDepth=model_cfg["maxDepth"],
    )

    pipeline = Pipeline(stages=indexers + [assembler, rf])
    model = pipeline.fit(df)

    # Save model
    model.write().overwrite().save("models/titanic_model")

    print("Titanic model trained and saved at models/titanic_model")
    return model, df, features_cfg["label"]


def train_mnist(spark, train_path, val_path, params):
    df = spark.read.parquet(train_path)
    label_col = params["features"]["mnist"]["label"]
    model_cfg = params["model"]["mnist"]

    feature_cols = [c for c in df.columns if c != label_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=model_cfg["maxIter"])
    pipeline = Pipeline(stages=[assembler, lr])
    model = pipeline.fit(df)

    model.write().overwrite().save("models/mnist_model")
    print("MNIST model trained and saved at models/mnist_model")
    return model, df, label_col


def main():
    args = parse_args()
    train_path = args.train
    val_path = args.val
    params_path = args.params

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    dataset_type = params["dataset"]["type"]
    spark = SparkSession.builder.appName("TrainingPipeline").getOrCreate()

    if dataset_type == "titanic":
        model, df, label = train_titanic(spark, train_path, val_path, params)
        evaluator = BinaryClassificationEvaluator(labelCol=label)
        auc = evaluator.evaluate(model.transform(df))
        print(f"Training AUC: {auc:.4f}")
        metrics = {"AUC": auc}

    elif dataset_type == "mnist":
        model, df, label = train_mnist(spark, train_path, val_path, params)
        evaluator = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")
        acc = evaluator.evaluate(model.transform(df))
        print(f"Training Accuracy: {acc:.4f}")
        metrics = {"accuracy": acc}

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    spark.stop()


if __name__ == "__main__":
    main()
