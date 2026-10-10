
from pathlib import Path
import json
import joblib
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.keras"
SCALER_PATH = BASE_DIR / "models" / "iris_scaler.pkl"

REPORT_PATH = BASE_DIR / "models" / "validation_report.json"
MIN_ACCURACY = 0.90

# Check that the required files exist
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError(
        "Model or scaler is missing. Run src/train.py first."
    )

# Load the model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Use a fixed test split for repeatable validation
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.20,
    random_state=42,
    stratify=iris.target
)

X_test_scaled = scaler.transform(X_test)

# Evaluate the trained model
probabilities = model.predict(X_test_scaled, verbose=0)
predictions = np.argmax(probabilities, axis=1)
accuracy = accuracy_score(y_test, predictions)

passed = accuracy >= MIN_ACCURACY

report = {
    "model": "Iris Neural Network",
    "accuracy": round(float(accuracy), 4),
    "minimum_accuracy": MIN_ACCURACY,
    "validation_status": "PASSED" if passed else "FAILED"
}

REPORT_PATH.write_text(json.dumps(report, indent=4))

print("Model accuracy:", round(accuracy * 100, 2), "%")
print("Minimum required:", MIN_ACCURACY * 100, "%")
print("Validation status:", report["validation_status"])
print("Report saved to:", REPORT_PATH)

if not passed:
    raise SystemExit("Model rejected: accuracy is below the threshold.")
