"""
features.py — Feature engineering pipeline for predictive maintenance.

This module transforms raw sensor readings into powerful features
that capture engine degradation patterns over time.

Feature categories:
  1. Rolling statistics (mean, std over windows of 5, 10, 20 cycles)
  2. Difference features (cycle-over-cycle rate of change)
  3. Exponential moving averages (recent readings weighted more)
  4. Time features (normalized cycle position)

WHY THESE FEATURES:
A single sensor reading at one moment tells you little. But the
TREND — how readings change over 5, 10, 20 cycles — reveals
whether the engine is degrading and how fast. This is exactly
how real predictive maintenance systems work at companies like
Siemens and Bosch.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Sensor selection (from EDA analysis) ──
# These sensors show clear degradation trends.
# Sensors 1, 5, 6, 10, 16, 18, 19 have near-zero variance → useless.
USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

OPERATIONAL_SETTINGS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# Top sensors most correlated with failure (from correlation analysis)
TOP_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11',
               'sensor_15', 'sensor_20', 'sensor_21']


def add_rolling_features(df: pd.DataFrame, window_sizes: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Add rolling mean and std for each useful sensor.
    
    Rolling statistics capture the TREND in sensor readings.
    - Rolling mean smooths out noise → shows underlying trend
    - Rolling std captures volatility → increases as engine degrades
    """
    df = df.copy()

    for sensor in USEFUL_SENSORS:
        for window in window_sizes:
            grouped = df.groupby('engine_id')[sensor]

            df[f'{sensor}_rmean_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{sensor}_rstd_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )

    return df


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cycle-over-cycle change for key sensors.
    
    The RATE of change matters: an engine where sensor values
    are changing rapidly is degrading faster than one with slow changes.
    """
    df = df.copy()

    for sensor in USEFUL_SENSORS:
        df[f'{sensor}_diff'] = df.groupby('engine_id')[sensor].diff().fillna(0)

    return df


def add_ema_features(df: pd.DataFrame, spans: list = [5, 10]) -> pd.DataFrame:
    """
    Add Exponential Moving Averages (EMA).
    
    EMA gives MORE weight to recent readings (unlike simple rolling mean
    which weights all equally). This makes it more responsive to
    sudden degradation — critical for catching rapid failure modes.
    """
    df = df.copy()

    for sensor in TOP_SENSORS:
        for span in spans:
            df[f'{sensor}_ema_{span}'] = df.groupby('engine_id')[sensor].transform(
                lambda x: x.ewm(span=span, min_periods=1).mean()
            )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized time/cycle features.
    
    cycle_norm = current_cycle / max_recorded_cycle
    This gives a 0→1 "how far along" indicator that helps
    the model understand temporal context.
    """
    df = df.copy()

    max_cycle_per_engine = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_norm'] = df['cycle'] / max_cycle_per_engine

    return df


def normalize_features(df: pd.DataFrame, feature_cols: list,
                       scaler: StandardScaler = None, fit: bool = True):
    """
    Normalize features to zero mean and unit variance.
    
    WHY: Different sensors have very different scales
    (e.g., sensor_9 ~9000 vs sensor_15 ~8.4). Without normalization,
    the model would be biased toward high-magnitude sensors.
    
    Returns: (normalized_df, fitted_scaler)
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def build_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    
    Call this on both training and test data to ensure
    consistent feature transformation.
    """
    print("  Building features...")

    # Keep only useful columns
    keep_cols = ['engine_id', 'cycle'] + OPERATIONAL_SETTINGS + USEFUL_SENSORS
    if 'RUL' in df.columns:
        keep_cols.append('RUL')

    # Handle case where some sensors might not exist (e.g., synthetic data)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Add engineered features
    print("    → Rolling statistics (window=5,10,20)...")
    df = add_rolling_features(df, window_sizes=[5, 10, 20])

    print("    → Cycle-over-cycle differences...")
    df = add_difference_features(df)

    print("    → Exponential moving averages...")
    df = add_ema_features(df)

    print("    → Time features...")
    df = add_time_features(df)

    # Fill any remaining NaN values
    df = df.fillna(0)

    print(f"    → Done! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns (exclude metadata and target)."""
    exclude = ['engine_id', 'cycle', 'RUL']
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from data_loader import load_train_data, cap_rul

    df = load_train_data()
    df = cap_rul(df)
    featured_df = build_features(df)

    feature_cols = get_feature_columns(featured_df)
    print(f"\n  Original columns:    26")
    print(f"  Engineered features: {len(feature_cols)}")
    print(f"\n  Sample feature names:")
    for col in feature_cols[:20]:
        print(f"    • {col}")
    print(f"    ... and {len(feature_cols) - 20} more")
