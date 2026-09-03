"""
src/lead_time_evaluator.py
--------------------------
STAGE 2 (Feature 2) — Warning Lead Time Empirical Evaluator.

Measures the advance warning lead time provided by the Random Forest model
on the chronological test set.

Warning Lead Time Definition:
    warning_lead_time = actual_congestion_time - first_high_risk_prediction_time

Strict Data Integrity:
- Computed exclusively on the chronological test split (last 15% of time series).
- Strict causal ordering: prediction at time t uses ONLY observations up to time t.
- Contiguous pre-congestion search: only searches back through contiguous normal-flow
  periods immediately preceding congestion onset (does not bridge across prior congested episodes).
- Identifies real transition events where traffic shifts from normal (>=20 mph)
  to congested (<20 mph).
- Calculates empirical metrics: mean, median, advance warning rate, and distribution.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import compute_spatial_features, SPATIAL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("lead_time_evaluator")


def evaluate_warning_lead_time(
    df_clean: pd.DataFrame = None,
    model_path: str = config.SPATIAL_MODEL_PKL,
    graph_path: str = config.ADJACENCY_PKL,
    save_path: str = config.LEAD_TIME_METRICS_JSON,
    risk_threshold_prob: float = 0.50,
    max_lookback_steps: int = 12, # 12 steps = 60 minutes lookback
) -> dict:
    """
    Evaluate warning lead time on historical chronological test set using fast batch vectorization
    and contiguous pre-congestion episode boundary checks.
    """
    log.info("Starting Warning Lead Time evaluation on test set...")
    
    if df_clean is None:
        df_raw = load_speed_data()
        df_clean = clean_speed_matrix(df_raw)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
        
    sensor_ids = list(df_clean.columns)
    
    # 1. Feature extraction
    temp_feats = extract_temporal_features(df_clean)
    sp_feats = compute_spatial_features(df_clean, G)
    
    # 2. Chronological test slice (last 15%)
    N_total = len(df_clean)
    test_start_idx = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    test_timestamps = df_clean.index[test_start_idx:]
    T_test = len(test_timestamps)
    N_sensors = len(sensor_ids)
    
    log.info("Evaluating %d test timestamps across %d sensors (%d total sensor-timestamps)...",
             T_test, N_sensors, T_test * N_sensors)
    
    # 3. Vectorized batch feature matrix construction
    rows_temp = []
    rows_spat = []
    
    for t in test_timestamps:
        t_feats = np.column_stack([
            temp_feats[name].loc[t].values if isinstance(temp_feats[name], pd.DataFrame)
            else np.full(N_sensors, temp_feats[name].loc[t])
            for name in TEMPORAL_FEATURE_NAMES
        ])
        s_feats = np.column_stack([
            sp_feats[name].loc[t].values
            for name in SPATIAL_FEATURE_NAMES
        ])
        rows_temp.append(t_feats)
        rows_spat.append(s_feats)
        
    X_test_temp = np.vstack(rows_temp).astype(np.float32)
    X_test_spat = np.vstack(rows_spat).astype(np.float32)
    X_test_all = np.hstack([X_test_temp, X_test_spat])
    
    log.info("Running batch model inference on %s test matrix...", X_test_all.shape)
    probs_flat = model.predict_proba(X_test_all)[:, 1]
    
    # Reshape back to (T_test, N_sensors)
    probs_matrix = probs_flat.reshape(T_test, N_sensors)
    test_speeds = df_clean.loc[test_timestamps, sensor_ids].values
    
    log.info("Inference complete. Calculating lead time across contiguous congestion onset events...")
    
    congestion_threshold = config.CONGESTION_THRESHOLD_MPH
    lead_times = [] # in minutes
    event_details = []
    total_events = 0
    advance_warned_count = 0
    same_step_warned_count = 0
    missed_count = 0
    
    # Scan through each sensor timeline
    for col_idx, s in enumerate(sensor_ids):
        s_speeds = test_speeds[:, col_idx]
        s_probs = probs_matrix[:, col_idx]
        
        for i in range(1, T_test):
            curr_spd = s_speeds[i]
            prev_spd = s_speeds[i - 1]
            
            # Transition: normal (>=20 mph) -> congested (<20 mph)
            if curr_spd < congestion_threshold and prev_spd >= congestion_threshold:
                total_events += 1
                t_event = test_timestamps[i]
                
                # Scan backwards ONLY through contiguous normal-flow period immediately prior to onset
                # (stop if hitting a prior congested episode)
                k = i - 1
                first_warn_idx = None
                steps_back = 0
                
                while k >= 0 and steps_back < max_lookback_steps:
                    if s_speeds[k] < congestion_threshold:
                        # Prior congestion boundary reached; stop lookback
                        break
                    if s_probs[k] >= risk_threshold_prob:
                        first_warn_idx = k
                    steps_back += 1
                    k -= 1
                    
                if first_warn_idx is not None:
                    lead_min = (i - first_warn_idx) * config.SAMPLE_INTERVAL_MIN
                    lead_times.append(lead_min)
                    advance_warned_count += 1
                    
                    if len(event_details) < 40:
                        event_details.append({
                            "sensor_id": str(s),
                            "congestion_timestamp": str(t_event),
                            "speed_at_onset_mph": round(float(curr_spd), 1),
                            "first_warning_timestamp": str(test_timestamps[first_warn_idx]),
                            "predicted_probability": round(float(s_probs[first_warn_idx]), 3),
                            "lead_time_minutes": int(lead_min),
                            "status": "Advance Warning",
                        })
                elif s_probs[i] >= risk_threshold_prob:
                    lead_times.append(0)
                    same_step_warned_count += 1
                    if len(event_details) < 40:
                        event_details.append({
                            "sensor_id": str(s),
                            "congestion_timestamp": str(t_event),
                            "speed_at_onset_mph": round(float(curr_spd), 1),
                            "first_warning_timestamp": str(t_event),
                            "predicted_probability": round(float(s_probs[i]), 3),
                            "lead_time_minutes": 0,
                            "status": "Same-Step Warning",
                        })
                else:
                    missed_count += 1
                    if len(event_details) < 40:
                        event_details.append({
                            "sensor_id": str(s),
                            "congestion_timestamp": str(t_event),
                            "speed_at_onset_mph": round(float(curr_spd), 1),
                            "first_warning_timestamp": None,
                            "predicted_probability": round(float(s_probs[i]), 3),
                            "lead_time_minutes": 0,
                            "status": "Missed Warning",
                        })
                        
    lead_times_arr = np.array(lead_times)
    adv_leads = lead_times_arr[lead_times_arr > 0]
    
    mean_lead = float(np.mean(adv_leads)) if len(adv_leads) > 0 else 0.0
    median_lead = float(np.median(adv_leads)) if len(adv_leads) > 0 else 0.0
    max_lead = int(np.max(adv_leads)) if len(adv_leads) > 0 else 0
    p25_lead = float(np.percentile(adv_leads, 25)) if len(adv_leads) > 0 else 0.0
    p75_lead = float(np.percentile(adv_leads, 75)) if len(adv_leads) > 0 else 0.0
    
    dist_5m = int(np.sum(lead_times_arr == 5))
    dist_10m = int(np.sum(lead_times_arr == 10))
    dist_15m = int(np.sum(lead_times_arr == 15))
    dist_20m = int(np.sum(lead_times_arr == 20))
    dist_25m = int(np.sum(lead_times_arr == 25))
    dist_30p = int(np.sum(lead_times_arr >= 30))
    
    adv_pct = round(100.0 * advance_warned_count / max(1, total_events), 2)
    total_warn_pct = round(100.0 * (advance_warned_count + same_step_warned_count) / max(1, total_events), 2)
    
    metrics = {
        "metadata": {
            "title": "TrafficFlow AI — Warning Lead Time Historical Test Benchmark",
            "dataset_split": "Chronological Test Set (Last 15% of Indian Driving Dataset (IDD))",
            "test_timesteps_evaluated": T_test,
            "total_sensors": N_sensors,
            "congestion_threshold_mph": config.CONGESTION_THRESHOLD_MPH,
            "risk_threshold_prob": risk_threshold_prob,
            "max_lookback_window_min": max_lookback_steps * config.SAMPLE_INTERVAL_MIN,
            "evaluation_principle": "Contiguous pre-congestion episode search without bridging prior congested periods",
            "generated_at": datetime.now().isoformat(),
        },
        "summary": {
            "total_congestion_events": total_events,
            "advance_warned_events": advance_warned_count,
            "same_step_warned_events": same_step_warned_count,
            "missed_events": missed_count,
            "advance_warning_rate_pct": adv_pct,
            "total_detection_rate_pct": total_warn_pct,
            "mean_lead_time_minutes": round(mean_lead, 1),
            "median_lead_time_minutes": round(median_lead, 1),
            "iqr_lead_time_minutes": [round(p25_lead, 1), round(p75_lead, 1)],
            "max_lead_time_minutes": max_lead,
        },
        "lead_time_distribution": {
            "5_min": {"count": dist_5m, "pct": round(100.0 * dist_5m / max(1, total_events), 1)},
            "10_min": {"count": dist_10m, "pct": round(100.0 * dist_10m / max(1, total_events), 1)},
            "15_min": {"count": dist_15m, "pct": round(100.0 * dist_15m / max(1, total_events), 1)},
            "20_min": {"count": dist_20m, "pct": round(100.0 * dist_20m / max(1, total_events), 1)},
            "25_min": {"count": dist_25m, "pct": round(100.0 * dist_25m / max(1, total_events), 1)},
            "30_plus_min": {"count": dist_30p, "pct": round(100.0 * dist_30p / max(1, total_events), 1)},
        },
        "sample_events": event_details,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    log.info(
        "✓ Saved warning lead time metrics to %s: Total Events=%d | Advance Warned=%d (%.1f%%) | Mean Lead=%.1f min | Median Lead=%.1f min",
        save_path, total_events, advance_warned_count, adv_pct, mean_lead, median_lead
    )
    return metrics


if __name__ == "__main__":
    evaluate_warning_lead_time()
