"""
predict.py — Load trained model and make RUL predictions.

Used by both the API and Dashboard to serve predictions.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path


class RULPredictor:
    """Loads the trained model and predicts Remaining Useful Life."""

    def __init__(self, model_path: str = "models/best_model.pkl",
                 features_path: str = "models/feature_columns.json"):
        """Load model and feature list from disk."""
        self.model = joblib.load(model_path)

        with open(features_path, "r") as f:
            self.feature_columns = json.load(f)

        print(f"  ✅ Model loaded ({len(self.feature_columns)} features expected)")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for a DataFrame of feature rows.
        Returns array of predicted RUL values, clipped to [0, 125].
        """
        # Ensure correct column order
        available_cols = [c for c in self.feature_columns if c in features_df.columns]
        X = features_df[available_cols]

        # Fill missing columns with 0
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 125)

    def predict_single(self, feature_dict: dict) -> float:
        """Predict RUL for a single observation (used by API)."""
        df = pd.DataFrame([feature_dict])
        return float(self.predict(df)[0])

    def get_health_status(self, rul: float) -> dict:
        """
        Convert numeric RUL into actionable health status.
        
        Thresholds are based on typical maintenance scheduling:
          > 80 cycles → HEALTHY  (no action needed)
          > 40 cycles → WARNING  (plan maintenance)
          > 15 cycles → CRITICAL (maintain NOW)
          ≤ 15 cycles → DANGER   (stop the engine!)
        """
        if rul > 80:
            return {
                "status": "HEALTHY",
                "color": "#27ae60",
                "emoji": "🟢",
                "action": "Continue normal operation. No maintenance needed.",
                "urgency": 1
            }
        elif rul > 40:
            return {
                "status": "WARNING",
                "color": "#f39c12",
                "emoji": "🟡",
                "action": "Schedule maintenance within the next 40 cycles.",
                "urgency": 2
            }
        elif rul > 15:
            return {
                "status": "CRITICAL",
                "color": "#e74c3c",
                "emoji": "🔴",
                "action": "Immediate maintenance required. Prepare replacement parts.",
                "urgency": 3
            }
        else:
            return {
                "status": "DANGER",
                "color": "#8b0000",
                "emoji": "🚨",
                "action": "STOP ENGINE IMMEDIATELY — failure is imminent!",
                "urgency": 4
            }


if __name__ == "__main__":
    predictor = RULPredictor()

    # Test health status logic
    for rul in [100, 60, 25, 10, 0]:
        health = predictor.get_health_status(rul)
        print(f"  RUL={rul:>3} → {health['emoji']} {health['status']:>8} | {health['action']}")
