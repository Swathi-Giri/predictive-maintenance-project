"""
data_loader.py — Loads and preprocesses the NASA C-MAPSS dataset.

The C-MAPSS dataset has NO headers. Each row has:
  engine_id | cycle | op_setting_1-3 | sensor_1-21

We load it, assign proper column names, and compute the target
variable RUL (Remaining Useful Life) for each row.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Column names (the raw files have no headers)
COLUMN_NAMES = [
    'engine_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]


def load_train_data(filepath: str = "data/raw/train_FD001.txt") -> pd.DataFrame:
    """
    Load the training data and compute RUL for each row.
    
    In the training set, each engine runs until failure.
    RUL = max_cycle_for_this_engine - current_cycle
    So at the last cycle, RUL = 0 (engine just failed).
    """
    df = pd.read_csv(
        filepath,
        sep=r'\s+',       # whitespace-separated
        header=None,       # no header row
        names=COLUMN_NAMES
    )

    # Make engine_id integer
    df['engine_id'] = df['engine_id'].astype(int)
    df['cycle'] = df['cycle'].astype(int)

    # Compute RUL: for each engine, max_cycle - current_cycle
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)

    return df


def load_test_data(
    test_filepath: str = "data/raw/test_FD001.txt",
    rul_filepath: str = "data/raw/RUL_FD001.txt"
) -> pd.DataFrame:
    """
    Load test data and attach true RUL values.
    
    Test data is cut off BEFORE failure. The true RUL file gives
    the remaining cycles at the LAST recorded cycle of each engine.
    For earlier cycles: RUL = true_rul + (max_recorded_cycle - current_cycle)
    """
    df = pd.read_csv(test_filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)
    df['engine_id'] = df['engine_id'].astype(int)
    df['cycle'] = df['cycle'].astype(int)

    # True RUL for the last cycle of each engine
    rul_true = pd.read_csv(rul_filepath, sep=r'\s+', header=None, names=['true_rul'])
    rul_true['engine_id'] = rul_true.index + 1

    # Compute RUL for all cycles
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']

    df = df.merge(max_cycles, on='engine_id')
    df = df.merge(rul_true, on='engine_id')
    df['RUL'] = df['true_rul'] + (df['max_cycle'] - df['cycle'])
    df.drop(['max_cycle', 'true_rul'], axis=1, inplace=True)

    return df


def cap_rul(df: pd.DataFrame, cap_value: int = 125) -> pd.DataFrame:
    """
    Cap the RUL at a maximum value (default 125).
    
    WHY THIS MATTERS:
    In early life, an engine is perfectly healthy. Whether it has
    200 or 300 cycles left, the health state is essentially the same.
    Capping at 125 helps the model focus on the DEGRADATION phase
    where predictions actually matter for maintenance decisions.
    
    This is a standard technique in predictive maintenance research.
    The value 125 comes from the original NASA C-MAPSS research papers.
    """
    df = df.copy()
    df['RUL'] = df['RUL'].clip(upper=cap_value)
    return df


if __name__ == "__main__":
    train_df = load_train_data()
    test_df = load_test_data()

    print("=" * 60)
    print("  NASA C-MAPSS DATASET SUMMARY")
    print("=" * 60)
    print(f"  Training: {train_df.shape[0]:,} rows | {train_df['engine_id'].nunique()} engines")
    print(f"  Testing:  {test_df.shape[0]:,} rows | {test_df['engine_id'].nunique()} engines")
    print(f"  Columns:  {train_df.shape[1]}")
    print(f"  RUL range (train): {train_df['RUL'].min()} - {train_df['RUL'].max()}")
    print(f"\n  First 3 rows:")
    print(train_df.head(3).to_string(index=False))
