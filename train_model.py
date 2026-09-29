import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("kidney_disease.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Clean target labels
df["classification"] = (
    df["classification"].astype(str).str.strip()
)

print("\nTarget distribution:")
print(df["classification"].value_counts())

# 3. Separate features and target
X = df.drop(columns=["classification", "id"])
y = df["classification"]

# 4. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Preprocess numerical columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Preprocess categorical columns
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 9. Create complete machine-learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 10. Train model
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed successfully!")

# 11. Evaluate model
y_pred = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Save complete pipeline
model_file = "kidney_disease_model.pkl"
joblib.dump(model, model_file)

print("\nModel saved:", model_file)
print("File exists:", os.path.exists(model_file))
print(f"File size: {os.path.getsize(model_file) / 1024:.2f} KB")
