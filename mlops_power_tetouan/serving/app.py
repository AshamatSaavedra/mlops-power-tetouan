# mlops_power_tetouan/serving/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
# from mlops_power_tetouan.config import set_seed

# set_seed()

app = FastAPI(title="Tetouan Energy Prediction", version="1.0")

MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"

# Cargar pipeline y modelos en memoria la primera vez
_feature_pipeline = None
_models_cache = {}

class PredictRequest(BaseModel):
    zone: str  # 'zone1' | 'zone2' | 'zone3'
    data: Dict[str, Any]  # keys: Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows, DateTime

@app.on_event("startup")
def load_artifacts():
    global _feature_pipeline
    if PIPELINE_PATH.exists():
        _feature_pipeline = joblib.load(PIPELINE_PATH)
    else:
        _feature_pipeline = None

def load_model_for_zone(zone: str):
    if zone in _models_cache:
        return _models_cache[zone]
    model_path = MODELS_DIR / f"{zone}_best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    model = joblib.load(model_path)
    _models_cache[zone] = model
    return model

@app.post("/predict")
def predict(req: PredictRequest):
    zone = req.zone
    if zone not in {"zone1", "zone2", "zone3"}:
        raise HTTPException(
            status_code=400,
            detail="zone must be one of zone1/zone2/zone3"
        )

    row = req.data.copy()

    # --- Fuerza DateTime a string siempre ---
    if "DateTime" in row:
        row["DateTime"] = str(row["DateTime"])

    df = pd.DataFrame([row])

    # --- DEBUG CRÍTICO ---
    print("DF QUE LLEGA AL PIPELINE:")
    print(df.dtypes)
    print(df)

    try:
        if _feature_pipeline is not None:
            X = _feature_pipeline.transform(df)
        else:
            X = df
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature pipeline error: {e}"
        )

    # -------- Model inference --------
    model = load_model_for_zone(zone)
    print("FEATURES DEL MODELO ENTRENADO:", model.feature_names_in_)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction error: {e}"
        )

    return {
        "zone": zone,
        "prediction": float(y_pred[0]),
        "model": str(type(model))
    }
