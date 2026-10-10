
import os
import random
import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "iris_model.keras"
SCALER_PATH = MODEL_DIR / "iris_scaler.pkl"

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# MLflow experiment
mlflow.set_experiment("Iris_Neural_Network")

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train and track
with mlflow.start_run() as run:
    mlflow.log_params({
        "model_type": "Neural Network",
        "hidden_layer_1": 16,
        "hidden_layer_2": 8,
        "epochs": 50,
        "batch_size": 8,
        "optimizer": "adam",
        "test_size": 0.2,
        "random_seed": SEED,
        "scaler": "StandardScaler"
    })

    model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=8,
        verbose=1
    )

    loss, _ = model.evaluate(
        X_test_scaled, y_test, verbose=0
    )

    probabilities = model.predict(
        X_test_scaled, verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("test_loss", float(loss))
    mlflow.log_metric("test_accuracy", float(accuracy))

    # Save deployment artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(SCALER_PATH))

    print("\nTest accuracy:", round(accuracy * 100, 2), "%")
    print("Test loss:", round(float(loss), 4))
    print("Run ID:", run.info.run_id)
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    ))

print("\nModel saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
