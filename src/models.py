# src/models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NaivePersistenceModel:
    """Baseline: predict last observed value."""
    def fit(self, y): self.last_value = y.iloc[-1]
    def predict(self, horizon): return np.full(horizon, self.last_value)


class MovingAverageModel:
    """Baseline: predict moving average of last k observations."""
    def __init__(self, window=7): self.window = window
    def fit(self, y): self.ma = y.iloc[-self.window:].mean()
    def predict(self, horizon): return np.full(horizon, self.ma)


class ARIMAForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
    
    def fit(self, y):
        try:
            self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order).fit()
        except Exception:
            self.model = ARIMA(y, order=(1, 1, 1)).fit()
        return self
    
    def predict(self, horizon):
        forecast = self.model.forecast(steps=horizon)
        return forecast.values if hasattr(forecast, 'values') else np.array(forecast)


class ExpSmoothingForecaster:
    def __init__(self, seasonal_periods=7):
        self.seasonal_periods = seasonal_periods
        self.model = None
    
    def fit(self, y):
        self.model = ExponentialSmoothing(
            y, trend='add', seasonal='add', seasonal_periods=self.seasonal_periods
        ).fit()
        return self
    
    def predict(self, horizon):
        return self.model.forecast(horizon).values


class MLForecaster:
    """Wrapper for Random Forest / Gradient Boosting with lag features."""
    def __init__(self, model_type='rf', n_lags=14, **kwargs):
        self.model_type = model_type
        self.n_lags = n_lags
        self.model = None
        self.kwargs = kwargs
        self.last_window = None
    
    def _make_supervised(self, y):
        X, Y = [], []
        for i in range(self.n_lags, len(y)):
            X.append(y.iloc[i - self.n_lags:i].values)
            Y.append(y.iloc[i])
        return np.array(X), np.array(Y)
    
    def fit(self, y):
        X, Y = self._make_supervised(y)
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 10),
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 3),
                random_state=42
            )
        self.model.fit(X, Y)
        self.last_window = y.iloc[-self.n_lags:].values.copy()
        return self
    
    def predict(self, horizon):
        preds = []
        window = self.last_window.copy()
        for _ in range(horizon):
            x = window.reshape(1, -1)
            p = self.model.predict(x)[0]
            preds.append(p)
            window = np.append(window[1:], p)
        return np.array(preds)