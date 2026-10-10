
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const result = document.getElementById("result");
    const button = document.getElementById("predictButton");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        result.hidden = false;
        result.className = "result";
        result.textContent = "Predicting flower species...";
        button.disabled = true;
        button.textContent = "Predicting...";

        const data = {
            sepal_length: Number(
                document.getElementById("sepal_length").value
            ),
            sepal_width: Number(
                document.getElementById("sepal_width").value
            ),
            petal_length: Number(
                document.getElementById("petal_length").value
            ),
            petal_width: Number(
                document.getElementById("petal_width").value
            )
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            const prediction = await response.json();

            if (!response.ok) {
                throw new Error(
                    prediction.detail
                        ? JSON.stringify(prediction.detail)
                        : "Prediction request failed."
                );
            }

            result.innerHTML = `
                <h2>Prediction Result</h2>
                <p><strong>Predicted Species:</strong>
                    ${prediction.predicted_species}</p>
                <p><strong>Confidence:</strong>
                    ${prediction.confidence}%</p>
                <p><strong>All Class Probabilities</strong></p>
                <p>Setosa: ${prediction.probabilities.Setosa}%</p>
                <p>Versicolor: ${prediction.probabilities.Versicolor}%</p>
                <p>Virginica: ${prediction.probabilities.Virginica}%</p>
            `;
        } catch (error) {
            result.className = "result error";
            result.textContent =
                "Prediction failed: " + error.message;
            console.error("Prediction error:", error);
        } finally {
            button.disabled = false;
            button.textContent = "Predict Flower";
        }
    });
});
