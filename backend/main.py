
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "iris_model.keras"
SCALER_PATH = MODEL_DIR / "iris_scaler.pkl"

# Check saved files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

# Load trained neural network
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: The scaler was saved using joblib.dump()
# Therefore, load it using joblib.load()
scaler = joblib.load(SCALER_PATH)

# Create FastAPI application
app = FastAPI(title="Iris Flower Prediction")

# Serve CSS and JavaScript
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)

class IrisData(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


class_names = ["Setosa", "Versicolor", "Virginica"]


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": True}


@app.post("/predict")
def predict(data: IrisData):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=float)

    # Apply the same scaler used during training
    scaled_features = scaler.transform(features)

    # Get prediction probabilities
    probabilities = model.predict(scaled_features, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_species": class_names[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
        "probabilities": {
            class_names[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(class_names))
        }
    }
