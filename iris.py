
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


# 1. Configure MLflow
mlflow.set_experiment("Iris_Neural_Network")


# 2. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Total data points:", len(X))
print("Number of input features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# 4. Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Create Neural Network
model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])


# 6. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Create models directory
os.makedirs("models", exist_ok=True)


# 7. Train, evaluate, and track using MLflow
with mlflow.start_run() as run:

    # Log parameters
    mlflow.log_param("model", "Neural Network")
    mlflow.log_param("hidden_layer_1", 16)
    mlflow.log_param("hidden_layer_2", 8)
    mlflow.log_param("output_classes", 3)
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 8)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 10)
    mlflow.log_param("scaler", "StandardScaler")

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=8,
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # Predict test samples
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate manual accuracy
    correct = np.sum(predicted_classes == y_test)
    total = len(y_test)
    manual_accuracy = correct / total

    # 8. Create evaluation table
    evaluation_table = pd.DataFrame({
        "actual_class": iris.target_names[y_test],
        "predicted_class": iris.target_names[predicted_classes],
        "actual_label": y_test,
        "predicted_label": predicted_classes,
        "correct": predicted_classes == y_test
    })

    # Log evaluation table to MLflow
    mlflow.log_table(
        data=evaluation_table,
        artifact_file="evaluation_results.json"
    )

    print("\nEvaluation table logged to MLflow!")

    # 9. Log metrics
    mlflow.log_metric("test_loss", float(loss))
    mlflow.log_metric("test_accuracy", float(accuracy))
    mlflow.log_metric("manual_accuracy", float(manual_accuracy))
    mlflow.log_metric("correct_predictions", int(correct))
    mlflow.log_metric("total_test_samples", int(total))

    # 10. Save trained model and scaler
    model_path = "models/iris_model.keras"
    scaler_path = "models/iris_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Log model and scaler as artifacts
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)

    # Print results
    print("\n-----------------------------")
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy * 100, "%")
    print("Correct Predictions:", correct)
    print("Total Test Samples:", total)
    print("Manual Accuracy:", manual_accuracy * 100, "%")
    print("MLflow Run ID:", run.info.run_id)
    print("-----------------------------")

    print("\nPredicted Classes:")
    print(predicted_classes)

    print("\nActual Classes:")
    print(y_test)


# 11. Predict a new flower
new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

new_flower_scaled = scaler.transform(new_flower)

new_prediction = model.predict(
    new_flower_scaled,
    verbose=0
)

predicted_class = np.argmax(new_prediction)

print("\nNew Flower Prediction:")
print("Class:", predicted_class)
print("Flower:", iris.target_names[predicted_class])

print("\nModel and scaler saved successfully!")
