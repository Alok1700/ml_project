# src/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(y_true, y_pred, model_name='model'):
    return {
        'model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mape(y_true, y_pred)
    }


def walk_forward_validation(series, model_class, initial_train_size, horizon, step=1, **model_kwargs):
    """
    Walk-forward validation: expand training window, forecast horizon, roll forward.
    """
    results = []
    n = len(series)
    start = initial_train_size
    
    while start + horizon <= n:
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]
        
        model = model_class(**model_kwargs)
        model.fit(train)
        preds = model.predict(horizon)
        
        res = evaluate_forecast(test.values, preds)
        res['train_end'] = train.index[-1]
        res['horizon'] = horizon
        results.append(res)
        start += step
    
    return pd.DataFrame(results)


def forecast_stability_index(walk_forward_results):
    """Lower CV of MAE across folds = more stable."""
    return walk_forward_results['MAE'].std() / walk_forward_results['MAE'].mean()


def capacity_breach_probability(forecast_samples, capacity):
    """Probability forecast exceeds capacity threshold."""
    return (forecast_samples > capacity).mean()