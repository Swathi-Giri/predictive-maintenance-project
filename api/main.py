"""
main.py — FastAPI REST API for Engine RUL Predictions.

Endpoints:
  GET  /           → API info
  GET  /health     → Health check
  POST /predict    → Predict RUL from sensor readings

Run:
  uvicorn api.main:app --reload --port 8000
  Then open http://localhost:8000/docs for interactive API docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.predict import RULPredictor

# ── App setup ──
app = FastAPI(
    title="🔧 Engine RUL Prediction API",
    description=(
        "Predicts the Remaining Useful Life (RUL) of turbofan engines "
        "using sensor data. Part of an end-to-end predictive maintenance "
        "system built with NASA C-MAPSS data."
    ),
    version="1.0.0",
    contact={"name": "Your Name", "email": "your.email@example.com"}
)

# Allow requests from Streamlit dashboard and any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.json')

try:
    predictor = RULPredictor(model_path=MODEL_PATH, features_path=FEATURES_PATH)
    model_loaded = True
except Exception as e:
    print(f"⚠️  Model not loaded: {e}")
    model_loaded = False
    predictor = None


# ── Request/Response schemas ──
class SensorReading(BaseModel):
    """Sensor data for one engine at one time cycle."""
    engine_id: int = Field(1, description="Engine identifier")
    cycle: int = Field(100, description="Current operating cycle")
    op_setting_1: float = Field(0.0, description="Altitude")
    op_setting_2: float = Field(0.0, description="Mach number")
    op_setting_3: float = Field(100.0, description="Throttle angle")
    sensor_2: float = Field(642.0)
    sensor_3: float = Field(1590.0)
    sensor_4: float = Field(1400.0)
    sensor_7: float = Field(554.0)
    sensor_8: float = Field(2388.0)
    sensor_9: float = Field(9046.0)
    sensor_11: float = Field(47.5)
    sensor_12: float = Field(522.0)
    sensor_13: float = Field(2388.0)
    sensor_14: float = Field(8139.0)
    sensor_15: float = Field(8.42)
    sensor_17: float = Field(392.0)
    sensor_20: float = Field(39.0)
    sensor_21: float = Field(23.4)

    class Config:
        json_schema_extra = {
            "example": {
                "engine_id": 1,
                "cycle": 150,
                "op_setting_1": -0.0007,
                "op_setting_2": -0.0004,
                "op_setting_3": 100.0,
                "sensor_2": 648.5,
                "sensor_3": 1600.2,
                "sensor_4": 1418.3,
                "sensor_7": 556.1,
                "sensor_8": 2395.7,
                "sensor_9": 9080.5,
                "sensor_11": 48.8,
                "sensor_12": 527.3,
                "sensor_13": 2395.7,
                "sensor_14": 8165.0,
                "sensor_15": 8.6,
                "sensor_17": 400.5,
                "sensor_20": 37.2,
                "sensor_21": 22.1
            }
        }


class PredictionResponse(BaseModel):
    engine_id: int
    predicted_rul: float
    health_status: str
    health_emoji: str
    health_color: str
    recommended_action: str
    urgency_level: int


# ── Endpoints ──
@app.get("/")
def root():
    """API information and available endpoints."""
    return {
        "name": "Engine RUL Prediction API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health")
def health_check():
    """Check if the API and model are ready."""
    return {
        "api_status": "running",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(reading: SensorReading):
    """
    Predict Remaining Useful Life from sensor readings.
    
    Send current sensor values and get back:
    - Predicted RUL (cycles until failure)
    - Health status (HEALTHY / WARNING / CRITICAL / DANGER)
    - Recommended maintenance action
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python src/train.py' first."
        )

    try:
        features = reading.model_dump()
        rul = predictor.predict_single(features)
        health = predictor.get_health_status(rul)

        return PredictionResponse(
            engine_id=reading.engine_id,
            predicted_rul=round(rul, 1),
            health_status=health["status"],
            health_emoji=health["emoji"],
            health_color=health["color"],
            recommended_action=health["action"],
            urgency_level=health["urgency"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── Run directly ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
