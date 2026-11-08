"""Étape 2 : API FastAPI qui sert models/model.pkl."""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = [
    "culture", "soil_moisture_pct", "air_temp_c", "air_humidity_pct",
    "solar_radiation_wm2", "rainfall_mm", "hour",
]

app = FastAPI(title="MpambolyMivoatry - API irrigation")
model = joblib.load("models/model.pkl")


class Mesure(BaseModel):
    culture: str = Field(examples=["tomate"])
    soil_moisture_pct: float = Field(ge=0, le=100)
    air_temp_c: float
    air_humidity_pct: float = Field(ge=0, le=100)
    solar_radiation_wm2: float = Field(ge=0)
    rainfall_mm: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(m: Mesure):
    X = pd.DataFrame([m.model_dump()])[FEATURES]
    proba = float(model.predict_proba(X)[0][1])
    return {"irrigation_needed": proba >= 0.5, "probability": round(proba, 3)}
