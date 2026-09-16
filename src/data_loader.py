# src/data_loader.py
import pandas as pd
import numpy as np

def load_uac_data(filepath: str) -> pd.DataFrame:
    """
    Load and parse the UAC dataset.
    Handles comma-formatted numbers, date parsing, and removes empty rows.
    """
    df = pd.read_csv(filepath)
    
    # Remove fully empty rows
    df = df.dropna(how='all')
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Clean numeric columns (remove commas)
    numeric_cols = [
        'Children apprehended and placed in CBP custody*',
        'Children in CBP custody',
        'Children transferred out of CBP custody',
        'Children in HHS Care',
        'Children discharged from HHS Care'
    ]
    
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename columns for simplicity
    df = df.rename(columns={
        'Children apprehended and placed in CBP custody*': 'cbp_intake',
        'Children in CBP custody': 'cbp_care_load',
        'Children transferred out of CBP custody': 'cbp_transfers_out',
        'Children in HHS Care': 'hhs_care_load',
        'Children discharged from HHS Care': 'hhs_discharges'
    })
    
    # Sort by date ascending
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def create_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a continuous daily time index, interpolate missing days.
    """
    df = df.set_index('Date').sort_index()
    
    # Create full daily range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)
    df.index.name = 'Date'
    
    # Interpolate missing values (time-based)
    df = df.interpolate(method='time', limit_direction='both')
    
    # Round to integers (counts)
    for col in df.columns:
        df[col] = df[col].round().astype(int)
    
    return df