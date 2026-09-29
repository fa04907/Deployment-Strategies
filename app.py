
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Locate the saved model relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kidney_disease_model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Kidney Disease Prediction API",
    description="API for serving a trained machine-learning pipeline.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float | None = None
    bp: float | None = None
    sg: float | None = None
    al: float | None = None
    su: float | None = None
    rbc: str | None = None
    pc: str | None = None
    pcc: str | None = None
    ba: str | None = None
    bgr: float | None = None
    bu: float | None = None
    sc: float | None = None
    sod: float | None = None
    pot: float | None = None
    hemo: float | None = None
    pcv: float | None = None
    wc: float | None = None
    rc: float | None = None
    htn: str | None = None
    dm: str | None = None
    cad: str | None = None
    appet: str | None = None
    pe: str | None = None
    ane: str | None = None


@app.get("/")
def home():
    return {
        "message": "Kidney Disease Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: PatientData):
    try:
        patient = pd.DataFrame([data.model_dump()])
        prediction = model.predict(patient)[0]

        return {
            "prediction": str(prediction),
            "note": "Model output only; not a medical diagnosis.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc
