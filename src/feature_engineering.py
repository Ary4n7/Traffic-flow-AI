"""
src/feature_engineering.py
--------------------------
STAGE 2 — Temporal Feature Engineering.

Generates temporal lagged features and calendar indicators:
- current speed (speed_t)
- speed 5 minutes ago (speed_t_minus_5)
- speed 10 minutes ago (speed_t_minus_10)
- speed 15 minutes ago (speed_t_minus_15)
- speed change over 5 minutes (speed_delta_5)
- speed change over 10 minutes (speed_delta_10)
- rolling mean speed over 30 min (rolling_mean_30min)
- rolling speed trend over 30 min (rolling_trend_30min)
- hour of day (hour)
- day of week (day_of_week)
- weekend indicator (is_weekend)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import clean_speed_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("feature_engineering")


TEMPORAL_FEATURE_NAMES = [
    "speed_t",
    "speed_t_minus_5",
    "speed_t_minus_10",
    "speed_t_minus_15",
    "speed_delta_5",
    "speed_delta_10",
    "rolling_mean_30min",
    "rolling_trend_30min",
    "hour",
    "day_of_week",
    "is_weekend",
]


def extract_temporal_features(df_speeds: pd.DataFrame) -> dict:
    """
    Extract vectorized temporal feature matrices.
    
    Parameters
    ----------
    df_speeds : pd.DataFrame (T, N)
        Cleaned speeds across T timestamps and N sensors.
        
    Returns
    -------
    dict of str -> pd.DataFrame (T, N) or pd.Series (T,)
        Dictionary containing all temporal feature matrices aligned with df_speeds.
    """
    log.info("Generating temporal features for %d timesteps and %d sensors...", len(df_speeds), len(df_speeds.columns))
    
    # Lagged speeds (5, 10, 15 min -> 1, 2, 3 steps)
    s_t = df_speeds
    s_t5 = df_speeds.shift(1)
    s_t10 = df_speeds.shift(2)
    s_t15 = df_speeds.shift(3)
    
    # Deltas
    delta_5 = s_t - s_t5
    delta_10 = s_t - s_t10
    
    # 30-min rolling stats (6 steps)
    rolling_mean_30 = df_speeds.rolling(window=6, min_periods=1).mean()
    rolling_trend_30 = df_speeds.rolling(window=6, min_periods=1).apply(
        lambda x: (x[-1] - x[0]) if len(x) > 1 else 0.0, raw=True
    )
    
    # Calendar features
    timestamps = df_speeds.index
    hour = pd.Series(timestamps.hour, index=timestamps, dtype="int8")
    day_of_week = pd.Series(timestamps.dayofweek, index=timestamps, dtype="int8")
    is_weekend = pd.Series((timestamps.dayofweek >= 5).astype("int8"), index=timestamps, dtype="int8")
    
    features = {
        "speed_t": s_t,
        "speed_t_minus_5": s_t5,
        "speed_t_minus_10": s_t10,
        "speed_t_minus_15": s_t15,
        "speed_delta_5": delta_5,
        "speed_delta_10": delta_10,
        "rolling_mean_30min": rolling_mean_30,
        "rolling_trend_30min": rolling_trend_30,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }
    
    log.info("✓ Temporal features created successfully (%d features)", len(features))
    return features


def build_temporal_dataset_for_sensor(features: dict, sensor_id: str) -> pd.DataFrame:
    """Extract tabular temporal feature DataFrame for a single sensor."""
    df_out = pd.DataFrame(index=features["speed_t"].index)
    for feat in ["speed_t", "speed_t_minus_5", "speed_t_minus_10", "speed_t_minus_15",
                 "speed_delta_5", "speed_delta_10", "rolling_mean_30min", "rolling_trend_30min"]:
        df_out[feat] = features[feat][sensor_id].astype("float32")
        
    df_out["hour"] = features["hour"]
    df_out["day_of_week"] = features["day_of_week"]
    df_out["is_weekend"] = features["is_weekend"]
    
    # Drop warm-up rows (first 3 lag steps)
    return df_out.dropna()


if __name__ == "__main__":
    from src.data_loader import load_speed_data
    df_raw = load_speed_data()
    df_clean = clean_speed_matrix(df_raw)
    feats = extract_temporal_features(df_clean)
    sample_df = build_temporal_dataset_for_sensor(feats, df_clean.columns[0])
    print("\nSample temporal dataset for sensor", df_clean.columns[0], "shape:", sample_df.shape)
    print(sample_df.head())
