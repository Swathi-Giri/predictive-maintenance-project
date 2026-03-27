# %% [markdown]
# # 🔍 Exploratory Data Analysis — NASA C-MAPSS Turbofan Engine Dataset
#
# This notebook explores the engine sensor data to understand:
# 1. How long engines survive before failure
# 2. Which sensors show degradation patterns
# 3. Which sensors correlate with Remaining Useful Life (RUL)
# 4. Which sensors are useless (constant/no variance)

# %% Setup
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

from src.data_loader import load_train_data, load_test_data, cap_rul

print("✅ Libraries loaded!")

# %% Load Data
train_df = load_train_data()
test_df = load_test_data()

print("=" * 60)
print("  DATASET OVERVIEW")
print("=" * 60)
print(f"  Training:  {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"  Testing:   {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
print(f"  Engines:   {train_df['engine_id'].nunique()} (train) | {test_df['engine_id'].nunique()} (test)")
print(f"\n  RUL Range: {train_df['RUL'].min()} to {train_df['RUL'].max()}")
print(f"\n  First 5 rows:")
train_df.head()

# %% Engine Lifetime Distribution
engine_lifetimes = train_df.groupby('engine_id')['cycle'].max()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(engine_lifetimes, bins=20, edgecolor='black', alpha=0.7, color='#2196F3')
axes[0].axvline(engine_lifetimes.mean(), color='red', linestyle='--',
                label=f'Mean: {engine_lifetimes.mean():.0f} cycles')
axes[0].axvline(engine_lifetimes.median(), color='orange', linestyle='--',
                label=f'Median: {engine_lifetimes.median():.0f} cycles')
axes[0].set_xlabel('Total Cycles Before Failure')
axes[0].set_ylabel('Number of Engines')
axes[0].set_title('Engine Lifetime Distribution')
axes[0].legend()

axes[1].boxplot(engine_lifetimes, vert=True)
axes[1].set_ylabel('Total Cycles Before Failure')
axes[1].set_title('Engine Lifetime Spread')

plt.tight_layout()
plt.savefig('../data/processed/engine_lifetimes.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Shortest life: {engine_lifetimes.min()} cycles")
print(f"Longest life:  {engine_lifetimes.max()} cycles")
print(f"Average life:  {engine_lifetimes.mean():.1f} cycles")

# %% [markdown]
# ## Sensor Degradation Over Time
# This is THE key plot — shows how sensors change as engines approach failure.

# %% Sensor Degradation Visualization
sample_engines = np.random.choice(train_df['engine_id'].unique(), 5, replace=False)
sample_df = train_df[train_df['engine_id'].isin(sample_engines)]

important_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                     'sensor_11', 'sensor_12', 'sensor_15', 'sensor_20', 'sensor_21']

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, sensor in enumerate(important_sensors):
    ax = axes[idx]
    for engine_id in sample_engines:
        engine_data = sample_df[sample_df['engine_id'] == engine_id]
        ax.plot(engine_data['cycle'], engine_data[sensor],
                alpha=0.7, label=f'Engine {engine_id}')
    ax.set_title(f'{sensor}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')

axes[0].legend(fontsize=7, loc='upper left')
plt.suptitle('Sensor Degradation Over Time — Key Sensors',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/processed/sensor_degradation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Correlation with RUL
sensor_cols = [c for c in train_df.columns if 'sensor' in c]
corr_with_rul = train_df[sensor_cols + ['RUL']].corr()['RUL'].drop('RUL').sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr_with_rul.values]
corr_with_rul.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
ax.set_xlabel('Correlation with RUL')
ax.set_title('Which Sensors Predict Failure?')
ax.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('../data/processed/correlation_rul.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 sensors MOST correlated with failure (negative = degrades toward failure):")
print(corr_with_rul.head(5))

# %% Identify Useless Sensors
sensor_std = train_df[sensor_cols].std()
useless = sensor_std[sensor_std < 0.01]

print("Sensors with near-zero variance (REMOVE THESE):")
for s in useless.index:
    print(f"  {s}: std = {sensor_std[s]:.6f}")

useful = [s for s in sensor_cols if s not in useless.index]
print(f"\nKeeping {len(useful)} out of {len(sensor_cols)} sensors")

# %% RUL Distribution — Capped vs Uncapped
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['RUL'], bins=40, alpha=0.7, color='#3498db', edgecolor='black')
axes[0].set_title('Raw RUL Distribution')
axes[0].set_xlabel('Remaining Useful Life (cycles)')

capped_df = cap_rul(train_df, cap_value=125)
axes[1].hist(capped_df['RUL'], bins=40, alpha=0.7, color='#e74c3c', edgecolor='black')
axes[1].set_title('Capped RUL (max=125)')
axes[1].set_xlabel('Remaining Useful Life (cycles)')

plt.suptitle('Why We Cap RUL: Focuses Model on Degradation Phase',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/rul_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Key Findings
#
# 1. **Engine lifetimes vary** from ~128 to ~362 cycles
# 2. **Sensors 2, 3, 4, 7, 11, 12, 15, 20, 21** show clear degradation trends
# 3. **Sensors 1, 5, 6, 10, 16, 18, 19** are near-constant → removed
# 4. **Capping RUL at 125** focuses the model on the degradation phase
# 5. Sensors with **negative correlation** degrade as failure approaches
#
# → These insights drive our feature engineering and model design.
