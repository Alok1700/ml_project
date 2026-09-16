# notebooks/eda_forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')

from src.data_loader import load_uac_data, create_daily_index
from src.preprocessing import decompose_series, summary_statistics, check_stationarity
from src.feature_engineering import build_feature_matrix
from src.models import *
from src.evaluation import *

# Load data
df = load_uac_data('../data/abc.csv')
df = create_daily_index(df)

print("Shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("\nSummary:\n", summary_statistics(df))

# Plot all series
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
for ax, col in zip(axes, df.columns):
    ax.plot(df.index, df[col], color='steelblue')
    ax.set_title(col)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('all_series.png', dpi=100)
plt.show()

# Decompose HHS care load
decomp = decompose_series(df['hhs_care_load'], period=7)
decomp.plot()
plt.savefig('decomposition.png', dpi=100)
plt.show()

# Stationarity
print("\nStationarity (HHS Care Load):", check_stationarity(df['hhs_care_load']))

# Walk-forward validation
results = walk_forward_validation(
    df['hhs_care_load'],
    model_class=MLForecaster,
    initial_train_size=400,
    horizon=14,
    step=14,
    model_type='rf'
)
print("\nWalk-forward results:\n", results)
print("Stability Index:", forecast_stability_index(results))