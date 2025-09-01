import mlflow
from mlflow.pyfunc import load_model
from fastapi import FastAPI, HTTPException
import pandas as pd

# Ensure correct tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = mlflow.tracking.MlflowClient()
model_name = "TitanicModel"  # must match UI exactly

# Debug: list model
print("Models visible in registry:", client.get_registered_model("TitanicModel"))

# Get latest staged version
versions = client.get_latest_versions(model_name, stages=["Staging"])
if not versions:
    raise RuntimeError(f"No model found in Staging for {model_name}")
staging_version = versions[0].version

model_uri = f"models:/{model_name}/{staging_version}"
model = load_model(model_uri)

app = FastAPI(title="Titanic Model API")

@app.get("/")
def root():
    return {"message": "Titanic Model API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {
        "prediction": preds,
        "model_version": staging_version
    }
