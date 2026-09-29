import joblib
import pandas as pd
import time

# Load the saved model
model = joblib.load("../kidney_disease_model.pkl")

# Load dataset
df = pd.read_csv("../kidney_disease.csv")

# Remove target and ID columns
X = df.drop(columns=["classification", "id"])

print("Starting streaming prediction...\n")

# Simulate incoming records
for i in range(10):
    record = X.iloc[[i]]

    # Predict one incoming record
    prediction = model.predict(record)

    print(
        f"Incoming Record {i + 1} "
        f"-> Prediction: {prediction[0]}"
    )

    # Wait one second before next record
    time.sleep(1)

print("\nStreaming simulation completed.")
