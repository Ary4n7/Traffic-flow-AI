"""
src/temporal_regimes.py
-----------------------
STAGE 4 (P1) — Temporal Regime Robustness & Diurnal Traffic Dynamics Benchmark.

Evaluates system performance across standard highway operational regimes on the chronological TEST set:
1. Morning Peak (07:00 – 10:00) — Inbound commute shockwaves
2. Midday Inter-Peak (10:00 – 16:00) — Commercial and midday flow
3. Evening Peak (16:00 – 20:00) — Outbound commute bottleneck spillovers
4. Night Off-Peak (20:00 – 07:00) — Free-flow baseline
5. Weekday (Monday–Friday) vs Weekend (Saturday–Sunday)

Measures per regime:
- Base congestion occurrence rate
- Model Discrimination: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Uncertainty Calibration: Brier Score, Expected Calibration Error (ECE)
- Warning Efficacy: Advance Warning Rate, Mean Lead Time, Median Lead Time
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids, load_distances
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import build_sensor_graph, compute_spatial_features, SPATIAL_FEATURE_NAMES
from src.probability_calibration import compute_expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("temporal_regimes")


def run_temporal_regime_evaluation(
    model_path: str = config.CALIBRATED_MODEL_PKL,
    save_path: str = config.TEMPORAL_REGIME_JSON,
) -> dict:
    """Execute evaluation across diurnal and weekday/weekend temporal regimes."""
    log.info("Starting Temporal Regime Evaluation on chronological test set...")
    
    df_raw = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    df_clean = clean_speed_matrix(df_raw)
    
    G = build_sensor_graph(dist_df, sensor_ids)
    temp_feats = extract_temporal_features(df_clean)
    sp_feats = compute_spatial_features(df_clean, G)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    sensor_ids = list(df_clean.columns)
    N_sensors = len(sensor_ids)
    
    # Chronological test split (last 15%)
    N_total = len(df_clean)
    n_test = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    test_timestamps = df_clean.index[n_test:]
    T_test = len(test_timestamps)
    
    # Build complete test matrix
    rows_temp, rows_spat = [], []
    for t in test_timestamps:
        t_f = np.column_stack([
            temp_feats[name].loc[t].values if isinstance(temp_feats[name], pd.DataFrame)
            else np.full(N_sensors, temp_feats[name].loc[t])
            for name in TEMPORAL_FEATURE_NAMES
        ])
        s_f = np.column_stack([sp_feats[name].loc[t].values for name in SPATIAL_FEATURE_NAMES])
        rows_temp.append(t_f)
        rows_spat.append(s_f)
        
    X_test_all = np.hstack([np.vstack(rows_temp), np.vstack(rows_spat)]).astype(np.float32)
    
    # Target (t+1 < 20 mph)
    future_speeds = df_clean.shift(-1).loc[test_timestamps, sensor_ids].values
    y_test_mat = (future_speeds < config.CONGESTION_THRESHOLD_MPH).astype(np.int8)
    y_test_all = y_test_mat.flatten()
    
    # Predictions
    probs_flat = model.predict_proba(X_test_all)[:, 1]
    probs_mat = probs_flat.reshape(T_test, N_sensors)
    speeds_mat = df_clean.loc[test_timestamps, sensor_ids].values
    
    # Define regimes
    hours = test_timestamps.hour
    dow = test_timestamps.dayofweek
    
    regimes = {
        "morning_peak": {
            "name": "Morning Peak (07:00 – 10:00)",
            "mask": (hours >= 7) & (hours < 10) & (dow < 5),
            "description": "Inbound urban commute traffic with high recurring congestion",
        },
        "midday": {
            "name": "Midday Inter-Peak (10:00 – 16:00)",
            "mask": (hours >= 10) & (hours < 16),
            "description": "Commercial freight and non-commute inter-peak flow",
        },
        "evening_peak": {
            "name": "Evening Peak (16:00 – 20:00)",
            "mask": (hours >= 16) & (hours < 20) & (dow < 5),
            "description": "Outbound commuter dispersals with severe bottleneck spillover cascades",
        },
        "night_offpeak": {
            "name": "Night Off-Peak (20:00 – 07:00)",
            "mask": ((hours >= 20) | (hours < 7)),
            "description": "Free-flow highway conditions with isolated incident drops",
        },
        "weekday": {
            "name": "All Weekday Hours (Mon–Fri)",
            "mask": (dow < 5),
            "description": "Standard business days with cyclical rush-hour dynamics",
        },
        "weekend": {
            "name": "All Weekend Hours (Sat–Sun)",
            "mask": (dow >= 5),
            "description": "Leisure traffic patterns with irregular midday surges",
        },
    }
    
    results = {}
    
    for r_key, r_info in regimes.items():
        ts_mask = r_info["mask"]
        indices = np.where(ts_mask)[0]
        
        if len(indices) == 0:
            continue
            
        # Flattened slice for classification metrics
        y_slice = y_test_mat[indices].flatten()
        p_slice = probs_mat[indices].flatten()
        y_pred = (p_slice >= 0.50).astype(int)
        
        acc = accuracy_score(y_slice, y_pred)
        prec = precision_score(y_slice, y_pred, zero_division=0)
        rec = recall_score(y_slice, y_pred, zero_division=0)
        f1 = f1_score(y_slice, y_pred, zero_division=0)
        
        try:
            roc = roc_auc_score(y_slice, p_slice)
            pr = average_precision_score(y_slice, p_slice)
        except Exception:
            roc, pr = 0.5, 0.0
            
        brier = brier_score_loss(y_slice, p_slice)
        ece = compute_expected_calibration_error(y_slice, p_slice)
        
        # Lead time calculation within this regime
        tot_events = 0
        adv_events = 0
        same_events = 0
        lead_times = []
        
        # For lead time, scan events occurring at timestamps within this regime
        for col_idx in range(N_sensors):
            s_spd = speeds_mat[:, col_idx]
            s_pr = probs_mat[:, col_idx]
            
            for i in indices:
                if i > 0 and s_spd[i] < config.CONGESTION_THRESHOLD_MPH and s_spd[i - 1] >= config.CONGESTION_THRESHOLD_MPH:
                    tot_events += 1
                    k = i - 1
                    first_k = None
                    steps_b = 0
                    while k >= 0 and steps_b < 12:
                        if s_spd[k] < config.CONGESTION_THRESHOLD_MPH:
                            break
                        if s_pr[k] >= 0.50:
                            first_k = k
                        steps_b += 1
                        k -= 1
                        
                    if first_k is not None:
                        adv_events += 1
                        lead_times.append((i - first_k) * 5)
                    elif s_pr[i] >= 0.50:
                        same_events += 1
                        lead_times.append(0)
                        
        adv_arr = np.array([l for l in lead_times if l > 0])
        mean_lead = float(np.mean(adv_arr)) if len(adv_arr) > 0 else 0.0
        median_lead = float(np.median(adv_arr)) if len(adv_arr) > 0 else 0.0
        adv_rate = (adv_events / max(1, tot_events)) * 100.0
        
        cm = confusion_matrix(y_slice, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        reg_result = {
            "name": r_info["name"],
            "description": r_info["description"],
            "timesteps_count": int(len(indices)),
            "total_samples": int(len(y_slice)),
            "congestion_prevalence_pct": round(float(100.0 * np.mean(y_slice)), 2),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "brier_score": round(float(brier), 4),
            "ece": round(float(ece), 4),
            "false_positive_rate": round(float(fpr), 4),
            "total_congestion_events": tot_events,
            "advance_warned_events": adv_events,
            "advance_warning_rate_pct": round(float(adv_rate), 2),
            "mean_lead_time_min": round(float(mean_lead), 1),
            "median_lead_time_min": round(float(median_lead), 1),
        }
        results[r_key] = reg_result
        
        log.info(
            "  [%s]: F1=%.4f | Prec=%.4f | Rec=%.4f | ROC=%.4f | Brier=%.4f | AdvRate=%.1f%% | MeanLead=%.1f min (Congestion Prev: %.1f%%)",
            r_info["name"], f1, prec, rec, roc, brier, adv_rate, mean_lead, reg_result["congestion_prevalence_pct"]
        )
        
    artifact = {
        "metadata": {
            "title": "TrafficFlow AI — Temporal Regime Robustness Benchmarks",
            "dataset_split": "Chronological Test Set (Last 15% of Indian Driving Dataset (IDD))",
            "generated_at": datetime.now().isoformat(),
        },
        "regimes": results,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    log.info("✓ Saved temporal regime benchmarks to %s", save_path)
    return artifact


if __name__ == "__main__":
    run_temporal_regime_evaluation()
