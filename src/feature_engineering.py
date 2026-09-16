# src/feature_engineering.py
import pandas as pd
import numpy as np


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list = [1, 7, 14]) -> pd.DataFrame:
    """Create lag features for the target variable."""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: list = [7, 14]) -> pd.DataFrame:
    """Create rolling mean and variance features."""
    df = df.copy()
    for w in windows:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].rolling(window=w).mean()
        df[f'{target_col}_roll_std_{w}'] = df[target_col].rolling(window=w).std()
    return df


def create_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create flow-based signals: net pressure indicator."""
    df = df.copy()
    # Net pressure on HHS system
    df['net_pressure_hhs'] = df['cbp_transfers_out'] - df['hhs_discharges']
    # Net pressure on CBP system
    df['net_pressure_cbp'] = df['cbp_intake'] - df['cbp_transfers_out']
    # Total system pressure
    df['total_intake'] = df['cbp_intake']
    df['total_exits'] = df['hhs_discharges']
    df['system_balance'] = df['total_intake'] - df['total_exits']
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features."""
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = 'hhs_care_load') -> pd.DataFrame:
    """Build complete feature matrix for forecasting."""
    df = create_flow_features(df)
    df = create_calendar_features(df)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = df.dropna()
    return df