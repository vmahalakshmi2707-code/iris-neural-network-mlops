
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
import logging

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_monitoring")

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

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Create application
app = FastAPI(title="Iris Flower Prediction with Monitoring")

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)

class_names = ["Setosa", "Versicolor", "Virginica"]


class IrisData(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


# In-memory monitoring data
monitor_lock = Lock()

metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "prediction_errors": 0,
    "total_latency_ms": 0.0,
    "total_confidence": 0.0,
    "species_counts": Counter(),
    "recent_predictions": deque(maxlen=20),
}


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict")
def predict(data: IrisData):
    started = perf_counter()

    with monitor_lock:
        metrics["total_requests"] += 1

    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]], dtype=float)

        scaled_features = scaler.transform(features)
        probabilities = model.predict(
            scaled_features, verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))
        species = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        latency_ms = (perf_counter() - started) * 1000

        result = {
            "predicted_species": species,
            "confidence": round(confidence, 2),
            "probabilities": {
                class_names[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(class_names))
            },
        }

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_species": species,
            "confidence": round(confidence, 2),
            "latency_ms": round(latency_ms, 2),
        }

        with monitor_lock:
            metrics["successful_predictions"] += 1
            metrics["total_latency_ms"] += latency_ms
            metrics["total_confidence"] += confidence
            metrics["species_counts"][species] += 1
            metrics["recent_predictions"].append(event)

        logger.info(
            "Prediction successful | species=%s | confidence=%.2f%% | latency=%.2fms",
            species, confidence, latency_ms
        )

        return result

    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000

        with monitor_lock:
            metrics["prediction_errors"] += 1
            metrics["total_latency_ms"] += latency_ms

        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check application logs."
        ) from exc


@app.get("/metrics")
def get_metrics():
    # Take a consistent snapshot of monitoring values
    with monitor_lock:
        total = metrics["total_requests"]
        successful = metrics["successful_predictions"]
        errors = metrics["prediction_errors"]

        average_latency = (
            metrics["total_latency_ms"] / total if total else 0.0
        )
        average_confidence = (
            metrics["total_confidence"] / successful
            if successful else 0.0
        )

        return {
            "total_requests": total,
            "successful_predictions": successful,
            "prediction_errors": errors,
            "success_rate_percent": round(
                successful / total * 100, 2
            ) if total else 0.0,
            "average_latency_ms": round(average_latency, 2),
            "average_confidence_percent": round(
                average_confidence, 2
            ),
            "predictions_by_species": dict(metrics["species_counts"]),
            "recent_predictions": list(metrics["recent_predictions"]),
            "note": (
                "Metrics are held in memory and reset when the service restarts. "
                "Accuracy and model drift require actual labels or reference data."
            ),
        }
