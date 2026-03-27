"""
train.py — Train and compare multiple ML models for RUL prediction.

Trains Random Forest, XGBoost, and LightGBM. Evaluates each with
both statistical metrics (MAE, RMSE, R²) and business metrics
(% within ±15 cycles, late prediction rate). Saves the best model.

Usage:
    cd src
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent dir to path so we can run from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_train_data, load_test_data, cap_rul
from src.features import build_features, get_feature_columns


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate statistical and business metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Business metrics
    within_15 = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_30 = np.mean(np.abs(y_true - y_pred) <= 30) * 100
    late_preds = np.mean(y_pred > y_true) * 100  # predicted more life than actual → dangerous!

    print(f"\n  {'='*55}")
    print(f"  📊 {model_name}")
    print(f"  {'='*55}")
    print(f"  MAE  (avg error):        {mae:.2f} cycles")
    print(f"  RMSE:                    {rmse:.2f} cycles")
    print(f"  R² Score:                {r2:.4f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Within ±15 cycles:       {within_15:.1f}%")
    print(f"  Within ±30 cycles:       {within_30:.1f}%")
    print(f"  Late predictions (⚠️):    {late_preds:.1f}%")
    print(f"  → Average prediction is off by ~{mae:.0f} cycles")

    return {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "within_15_pct": round(within_15, 1),
        "within_30_pct": round(within_30, 1),
        "late_prediction_pct": round(late_preds, 1),
    }


def train_all_models():
    """Train 3 models, compare them, save the best one."""

    print("\n" + "🔧" * 30)
    print("  PREDICTIVE MAINTENANCE — MODEL TRAINING")
    print("🔧" * 30)

    # ── Step 1: Load data ──
    print("\n📥 Loading data...")
    train_df = load_train_data()
    test_df = load_test_data()

    print(f"  Train: {train_df.shape[0]:,} rows | {train_df['engine_id'].nunique()} engines")
    print(f"  Test:  {test_df.shape[0]:,} rows | {test_df['engine_id'].nunique()} engines")

    # ── Step 2: Cap RUL ──
    print("\n📐 Capping RUL at 125 cycles...")
    train_df = cap_rul(train_df, cap_value=125)
    test_df = cap_rul(test_df, cap_value=125)

    # ── Step 3: Feature engineering ──
    print("\n⚙️  Engineering features (training set)...")
    train_featured = build_features(train_df, is_training=True)

    print("\n⚙️  Engineering features (test set)...")
    test_featured = build_features(test_df, is_training=False)

    feature_cols = get_feature_columns(train_featured)

    X_train = train_featured[feature_cols].values
    y_train = train_featured['RUL'].values
    X_test = test_featured[feature_cols].values
    y_test = test_featured['RUL'].values

    print(f"\n  Feature count: {len(feature_cols)}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")

    # ── Step 4: Define models ──
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
    }

    # Try to import XGBoost and LightGBM
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    except ImportError:
        print("  ⚠️  XGBoost not installed — skipping")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    except ImportError:
        print("  ⚠️  LightGBM not installed — skipping")

    # ── Step 5: Train and evaluate ──
    print("\n" + "=" * 60)
    print("  🏋️ TRAINING MODELS")
    print("=" * 60)

    all_results = []
    best_mae = float('inf')
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 125)  # keep predictions in valid range

        metrics = evaluate_model(y_test, y_pred, name)
        all_results.append(metrics)

        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_model_name = name
            best_model = model

    # ── Step 6: Save the best model ──
    print(f"\n{'*'*60}")
    print(f"  🏆 BEST MODEL: {best_model_name}")
    print(f"  📉 MAE: {best_mae:.2f} cycles")
    print(f"{'*'*60}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Save model
    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n  💾 Model saved to: {model_path}")

    # Save feature column list (needed for deployment)
    features_path = model_dir / "feature_columns.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  💾 Feature list saved to: {features_path}")

    # Save comparison results
    results_path = model_dir / "model_comparison.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"  💾 Comparison saved to: {results_path}")

    print(f"\n  📊 Model Comparison:")
    print(results_df.to_string(index=False))

    print(f"\n  ✅ Training complete! Next steps:")
    print(f"     • Start API:       cd api && uvicorn main:app --reload")
    print(f"     • Start Dashboard: cd dashboard && streamlit run app.py")

    return best_model, feature_cols


if __name__ == "__main__":
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    best_model, features = train_all_models()
