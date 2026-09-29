import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../kidney_disease_model.pkl")

# Load the dataset
df = pd.read_csv("../kidney_disease.csv")

print("Number of records:", len(df))

# Keep IDs for identification
ids = df["id"]

# Remove target and ID columns
X = df.drop(columns=["classification", "id"])

# Generate predictions for all records
predictions = model.predict(X)

# Create output DataFrame
output = pd.DataFrame({
    "id": ids,
    "prediction": predictions
})

# Save predictions to CSV
output.to_csv("batch_predictions.csv", index=False)

print("\nBatch prediction completed!")
print("\nFirst 10 predictions:")
print(output.head(10))

print("\nOutput file created: batch_predictions.csv")
