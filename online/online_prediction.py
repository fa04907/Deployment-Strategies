import joblib
import pandas as pd

# Load the saved model
model = joblib.load("../kidney_disease_model.pkl")

# One sample for online prediction
data = {
    "age": 48,
    "bp": 80,
    "sg": 1.020,
    "al": 1,
    "su": 0,
    "rbc": "normal",
    "pc": "normal",
    "pcc": "notpresent",
    "ba": "notpresent",
    "bgr": 121,
    "bu": 36,
    "sc": 1.2,
    "sod": 138,
    "pot": 4.5,
    "hemo": 15.4,
    "pcv": "44",
    "wc": "7800",
    "rc": "5.2",
    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
}

# Convert the record into a DataFrame
input_df = pd.DataFrame([data])

# Predict
prediction = model.predict(input_df)

print("Input received:")
print(input_df)

print("\nOnline prediction:")
print(prediction[0])
