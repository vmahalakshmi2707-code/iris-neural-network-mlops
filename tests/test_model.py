
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.keras"
SCALER_PATH = BASE_DIR / "models" / "iris_scaler.pkl"


def test_model_and_scaler_exist():
    assert MODEL_PATH.exists()
    assert SCALER_PATH.exists()


def test_model_prediction_shape():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = scaler.transform(sample)

    probabilities = model.predict(sample_scaled, verbose=0)

    assert probabilities.shape == (1, 3)


def test_prediction_probabilities_are_valid():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = scaler.transform(sample)

    probabilities = model.predict(sample_scaled, verbose=0)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
