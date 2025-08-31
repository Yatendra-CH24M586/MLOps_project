# src/utils.py
import numpy as np
import struct


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols) / 255.0  # normalize
        return data


def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def prepare_mnist(images_path, labels_path, spark):
    X = load_mnist_images(images_path)
    y = load_mnist_labels(labels_path)

    data = np.hstack((X, y.reshape(-1, 1)))  # features + label
    cols = [f"pixel{i}" for i in range(X.shape[1])] + ["label"]

    df = spark.createDataFrame(data.tolist(), cols)
    return df
