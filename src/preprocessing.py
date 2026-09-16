# src/preprocessing.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose_series(series: pd.Series, period: int = 7):
    """Decompose a time series into trend, seasonal, and residual components."""
    series = series.asfreq('D').interpolate()
    decomposition = seasonal_decompose(series, model='additive', period=period)
    return decomposition


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for all numeric columns."""
    stats = df.describe().T
    stats['cv'] = stats['std'] / stats['mean']  # coefficient of variation
    return stats


def check_stationarity(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def detect_outliers(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Detect outliers using z-score method."""
    z_scores = (series - series.mean()) / series.std()
    return series[np.abs(z_scores) > z_thresh]