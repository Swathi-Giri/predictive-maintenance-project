"""
generate_sample_data.py — Generate realistic synthetic data matching NASA C-MAPSS format.

This creates sample data so you can test the entire pipeline even before
downloading the real dataset. The real dataset should replace these files.

The synthetic data mimics the key properties of C-MAPSS FD001:
- 100 engines with varying lifetimes (128-362 cycles)
- 3 operational settings + 21 sensor readings
- Sensors show degradation trends as engines approach failure
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def generate_engine_data(engine_id: int, max_cycles: int) -> np.ndarray:
    """Generate sensor data for one engine that degrades over time."""

    rows = []
    for cycle in range(1, max_cycles + 1):
        # Degradation factor: 0 (healthy) → 1 (failed)
        degradation = (cycle / max_cycles) ** 1.5  # nonlinear degradation

        # Operational settings (mostly constant with small noise)
        op1 = -0.0007 + np.random.normal(0, 0.001)
        op2 = -0.0004 + np.random.normal(0, 0.0005)
        op3 = 100.0 + np.random.normal(0, 0.01)

        # Sensor readings — some degrade, some stay constant, some are noisy
        # Constant sensors (no degradation info)
        s1 = 518.67 + np.random.normal(0, 0.5)
        s5 = 14.62 + np.random.normal(0, 0.1)
        s6 = 21.61 + np.random.normal(0, 0.3)
        s10 = 1.3 + np.random.normal(0, 0.01)
        s16 = 0.03 + np.random.normal(0, 0.001)
        s18 = 2388 + np.random.normal(0, 5)
        s19 = 100.0 + np.random.normal(0, 0.01)

        # Degrading sensors (these carry the useful signal)
        s2 = 642.15 + degradation * 10 + np.random.normal(0, 0.5)
        s3 = 1589.70 + degradation * 20 + np.random.normal(0, 1.0)
        s4 = 1400.60 + degradation * 30 + np.random.normal(0, 2.0)
        s7 = 553.75 + degradation * 5 + np.random.normal(0, 0.3)
        s8 = 2388.02 + degradation * 15 + np.random.normal(0, 1.0)
        s9 = 9046.19 + degradation * 50 + np.random.normal(0, 5.0)
        s11 = 47.47 + degradation * 2 + np.random.normal(0, 0.1)
        s12 = 521.66 + degradation * 8 + np.random.normal(0, 0.5)
        s13 = 2388.02 + degradation * 15 + np.random.normal(0, 1.0)
        s14 = 8138.62 + degradation * 40 + np.random.normal(0, 5.0)
        s15 = 8.4195 + degradation * 0.5 + np.random.normal(0, 0.02)
        s17 = 392 + degradation * 15 + np.random.normal(0, 1.0)
        s20 = 38.95 - degradation * 2 + np.random.normal(0, 0.1)
        s21 = 23.4190 - degradation * 1.5 + np.random.normal(0, 0.05)

        row = [
            engine_id, cycle, op1, op2, op3,
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
            s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21
        ]
        rows.append(row)

    return np.array(rows)


def main():
    print("🔧 Generating synthetic C-MAPSS-style data...")

    # Generate training data (100 engines, full run-to-failure)
    train_lifetimes = np.random.randint(128, 362, size=100)
    train_data = []

    for engine_id in range(1, 101):
        max_cycles = train_lifetimes[engine_id - 1]
        engine_rows = generate_engine_data(engine_id, max_cycles)
        train_data.append(engine_rows)

    train_array = np.vstack(train_data)

    # Save training data (space-separated, no header — matches NASA format)
    train_path = os.path.join(RAW_DIR, "train_FD001.txt")
    np.savetxt(train_path, train_array, fmt='%.4f', delimiter=' ')
    print(f"   ✅ train_FD001.txt — {train_array.shape[0]} rows, 100 engines")

    # Generate test data (100 engines, cut short before failure)
    test_lifetimes = np.random.randint(128, 362, size=100)
    test_data = []
    true_ruls = []

    for engine_id in range(1, 101):
        full_life = test_lifetimes[engine_id - 1]
        # Cut off at a random point (between 50% and 90% of life)
        cutoff_frac = np.random.uniform(0.5, 0.9)
        cutoff = int(full_life * cutoff_frac)

        engine_rows = generate_engine_data(engine_id, cutoff)
        test_data.append(engine_rows)

        # True RUL = remaining cycles from cutoff to failure
        true_ruls.append(full_life - cutoff)

    test_array = np.vstack(test_data)

    test_path = os.path.join(RAW_DIR, "test_FD001.txt")
    np.savetxt(test_path, test_array, fmt='%.4f', delimiter=' ')
    print(f"   ✅ test_FD001.txt  — {test_array.shape[0]} rows, 100 engines")

    # Save true RUL values
    rul_path = os.path.join(RAW_DIR, "RUL_FD001.txt")
    np.savetxt(rul_path, true_ruls, fmt='%d')
    print(f"   ✅ RUL_FD001.txt   — {len(true_ruls)} engines")

    print(f"\n📁 Files saved to {RAW_DIR}/")
    print(f"   Total training samples: {train_array.shape[0]:,}")
    print(f"   Total test samples:     {test_array.shape[0]:,}")
    print(f"   Average engine life:    {np.mean(train_lifetimes):.0f} cycles")
    print(f"\n⚠️  This is SYNTHETIC data for development.")
    print(f"   Replace with real NASA data: python download_data.py")


if __name__ == "__main__":
    main()
