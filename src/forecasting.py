# src/forecasting.py
import numpy as np
import pandas as pd


def recursive_forecast_with_ci(model, horizon, n_simulations=200, residual_std=None):
    """
    Generate forecast with confidence intervals via residual bootstrapping.
    """
    base_pred = model.predict(horizon)
    
    if residual_std is None:
        residual_std = np.std(base_pred) * 0.05
    
    sims = np.zeros((n_simulations, horizon))
    for i in range(n_simulations):
        noise = np.random.normal(0, residual_std, horizon)
        sims[i] = base_pred + np.cumsum(noise) * 0.3
    
    lower = np.percentile(sims, 5, axis=0)
    upper = np.percentile(sims, 95, axis=0)
    
    return base_pred, lower, upper, sims


def generate_forecast_report(series, horizon=14):
    """Run all models and return comparison."""
    from src.models import (
        NaivePersistenceModel, MovingAverageModel,
        ARIMAForecaster, ExpSmoothingForecaster, MLForecaster
    )
    
    models = {
        'Naive': NaivePersistenceModel(),
        'MovingAvg(7)': MovingAverageModel(window=7),
        'ARIMA': ARIMAForecaster(),
        'ExpSmoothing': ExpSmoothingForecaster(),
        'RandomForest': MLForecaster(model_type='rf'),
        'GradientBoosting': MLForecaster(model_type='gb')
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(series)
            preds = model.predict(horizon)
            results[name] = preds
        except Exception as e:
            print(f"{name} failed: {e}")
    
    return results