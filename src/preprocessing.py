"""
src/preprocessing.py
--------------------
Cleans and prepares raw speed time series.

Key Considerations:
- In METR-LA, 0.0 values often represent sensor dropouts/missing readings rather than true 0 mph stops.
- We replace isolated 0.0 values (sensor dropout) with forward fill (or time interpolation) up to a max gap.
- Long outage periods (> 1 hour) are left or handled cleanly to avoid fabricating data.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("preprocessing")


def clean_speed_matrix(df: pd.DataFrame, max_fill_gap_steps: int = 6) -> pd.DataFrame:
    """
    Clean speed measurements:
    1. Replace 0.0 values with NaN (temporary marker for dropouts).
    2. Linear interpolate / forward-fill gaps up to `max_fill_gap_steps` (e.g. 6 steps = 30 mins).
    3. Fill any remaining leading/trailing NaNs with column median or clamp to valid speed.
    4. Clip speeds to physically realistic range [0.0, 85.0] mph.
    """
    df_clean = df.copy()
    
    # Identify zeros as missing sensor readings
    zero_mask = (df_clean == 0.0)
    df_clean[zero_mask] = np.nan
    
    # Interpolate short gaps along time dimension
    df_clean = df_clean.interpolate(method="time", limit=max_fill_gap_steps)
    
    # Forward fill then backward fill any small remnants
    df_clean = df_clean.ffill(limit=max_fill_gap_steps).bfill(limit=max_fill_gap_steps)
    
    # For any prolonged outages remaining, fill with the sensor's historical median at that hour
    if df_clean.isna().sum().sum() > 0:
        log.info("Filling remaining long-gap dropouts with sensor-specific hourly medians...")
        hour = df_clean.index.hour
        for h in range(24):
            mask_h = (hour == h)
            hourly_median = df_clean[mask_h].median()
            df_clean[mask_h] = df_clean[mask_h].fillna(hourly_median)
            
        # Ultimate fallback if still NaN
        df_clean = df_clean.fillna(df_clean.median()).fillna(55.0)
        
    df_clean = df_clean.clip(lower=0.0, upper=85.0).astype("float32")
    
    log.info("Cleaned speed matrix: %s (zeros replaced/interpolated)", df_clean.shape)
    return df_clean
