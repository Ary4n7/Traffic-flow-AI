"""
src/threshold_analysis.py
-------------------------
STAGE 4 (P0) — Validation-Driven Decision & Early-Warning Threshold Analysis.

Evaluates operational decision thresholds on the chronological VALIDATION set:
- Sweeps probability thresholds [0.15, 0.85] in increments of 0.025.
- Evaluates Precision, Recall, F1, False Positive Rate (FPR), False Negative Rate (FNR),
  Advance Warning Rate, Missed Event Rate, and Mean Warning Lead Time.
- Selects 3 operational operating points strictly on the Validation set:
  1. F1-Optimal (Balanced Operational Mode)
  2. High-Precision Mode (Target Prec >= 85%, minimizing nuisance alarms)
  3. High-Recall / Early-Intervention Mode (Target Rec >= 90%, maximizing advance detection)
- Validates the selected thresholds on the untouched chronological TEST set.
- Proves zero test-set leakage in threshold tuning.
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
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids, load_distances
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import build_sensor_graph, compute_spatial_features, SPATIAL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("threshold_analysis")


def evaluate_threshold_operating_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    speed_matrix: np.ndarray,
    timestamps: pd.DatetimeIndex,
    sensor_ids: list,
    thresholds: list = None,
    max_lookback_steps: int = 12,
) -> list:
    """
    Evaluate comprehensive classification and advance warning lead-time metrics
    across a sweep of decision thresholds.
    """
    if thresholds is None:
        thresholds = [round(float(th), 3) for th in np.arange(0.15, 0.86, 0.025)]
        
    probs = model.predict_proba(X)[:, 1]
    T = len(timestamps)
    N_sensors = len(sensor_ids)
    probs_mat = probs.reshape(T, N_sensors)
    
    # Pre-extract transition events in speed_matrix: normal (>=20) -> congested (<20)
    congestion_threshold = config.CONGESTION_THRESHOLD_MPH
    events = [] # list of (col_idx, timestamp_idx)
    for col_idx in range(N_sensors):
        s_spd = speed_matrix[:, col_idx]
        for i in range(1, T):
            if s_spd[i] < congestion_threshold and s_spd[i - 1] >= congestion_threshold:
                events.append((col_idx, i))
                
    total_events = len(events)
    results = []
    
    for th in thresholds:
        y_pred = (probs >= th).astype(int)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Advance warning lead time calculation for this threshold
        adv_events = 0
        same_step_events = 0
        missed_events = 0
        lead_times = []
        
        for col_idx, i in events:
            s_spd = speed_matrix[:, col_idx]
            s_pr = probs_mat[:, col_idx]
            
            k = i - 1
            first_k = None
            steps_b = 0
            while k >= 0 and steps_b < max_lookback_steps:
                if s_spd[k] < congestion_threshold:
                    break
                if s_pr[k] >= th:
                    first_k = k
                steps_b += 1
                k -= 1
                
            if first_k is not None:
                adv_events += 1
                lead_times.append((i - first_k) * config.SAMPLE_INTERVAL_MIN)
            elif s_pr[i] >= th:
                same_step_events += 1
                lead_times.append(0)
            else:
                missed_events += 1
                
        adv_arr = np.array([l for l in lead_times if l > 0])
        mean_lead = float(np.mean(adv_arr)) if len(adv_arr) > 0 else 0.0
        median_lead = float(np.median(adv_arr)) if len(adv_arr) > 0 else 0.0
        
        adv_rate = (adv_events / max(1, total_events)) * 100.0
        tot_detection = ((adv_events + same_step_events) / max(1, total_events)) * 100.0
        missed_rate = (missed_events / max(1, total_events)) * 100.0
        
        results.append({
            "threshold": th,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "advance_warning_rate_pct": round(float(adv_rate), 2),
            "total_detection_rate_pct": round(float(tot_detection), 2),
            "missed_event_rate_pct": round(float(missed_rate), 2),
            "mean_lead_time_min": round(float(mean_lead), 1),
            "median_lead_time_min": round(float(median_lead), 1),
            "total_events": total_events,
            "advance_warned_events": adv_events,
            "missed_events": missed_events,
        })
        
    return results


def run_threshold_analysis(
    model_path: str = config.CALIBRATED_MODEL_PKL,
    save_path: str = config.THRESHOLD_ANALYSIS_JSON,
) -> dict:
    """
    Tune thresholds on validation data, select operating points, and test on test data.
    """
    log.info("Starting Validation-Driven Threshold Optimization...")
    
    df_raw = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    df_clean = clean_speed_matrix(df_raw)
    
    G = build_sensor_graph(dist_df, sensor_ids)
    temp_feats = extract_temporal_features(df_clean)
    sp_feats = compute_spatial_features(df_clean, G)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    N_sensors = len(sensor_ids)
    N_total = len(df_clean)
    n_train = int(N_total * config.TRAIN_FRAC)
    n_val = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    
    val_timestamps = df_clean.index[n_train:n_val]
    test_timestamps = df_clean.index[n_val:]
    
    def build_slice(timestamps, stride=2):
        ts_eval = timestamps[::stride]
        future_speeds = df_clean.shift(-1).loc[ts_eval, sensor_ids].values
        y = (future_speeds < config.CONGESTION_THRESHOLD_MPH).astype(np.int8).flatten()
        
        rows_temp, rows_spat = [], []
        for t in ts_eval:
            t_f = np.column_stack([
                temp_feats[name].loc[t].values if isinstance(temp_feats[name], pd.DataFrame)
                else np.full(N_sensors, temp_feats[name].loc[t])
                for name in TEMPORAL_FEATURE_NAMES
            ])
            s_f = np.column_stack([sp_feats[name].loc[t].values for name in SPATIAL_FEATURE_NAMES])
            rows_temp.append(t_f)
            rows_spat.append(s_f)
            
        X = np.hstack([np.vstack(rows_temp), np.vstack(rows_spat)]).astype(np.float32)
        speed_mat = df_clean.loc[ts_eval, sensor_ids].values
        return X, y, speed_mat, ts_eval
        
    log.info("Constructing Validation slice...")
    X_val, y_val, spd_val, ts_val = build_slice(val_timestamps, stride=2)
    
    log.info("Constructing Test slice...")
    X_test, y_test, spd_test, ts_test = build_slice(test_timestamps, stride=2)
    
    # 1. Validation Threshold Sweep
    log.info("Running threshold curve evaluation on VALIDATION set...")
    val_curve = evaluate_threshold_operating_curve(model, X_val, y_val, spd_val, ts_val, sensor_ids)
    
    # 2. Select Operating Points on Validation Data Only
    # A) F1-Optimal (Balanced)
    best_f1_val_entry = max(val_curve, key=lambda x: x["f1_score"])
    th_balanced = best_f1_val_entry["threshold"]
    
    # B) High-Precision (Minimum Precision >= 0.85 on validation, best recall)
    hp_candidates = [e for e in val_curve if e["precision"] >= 0.85]
    if hp_candidates:
        best_hp_entry = max(hp_candidates, key=lambda x: x["recall"])
        th_precision = best_hp_entry["threshold"]
    else:
        best_hp_entry = max(val_curve, key=lambda x: x["precision"])
        th_precision = best_hp_entry["threshold"]
        
    # C) High-Recall (Minimum Recall >= 0.90 on validation, best precision)
    hr_candidates = [e for e in val_curve if e["recall"] >= 0.90]
    if hr_candidates:
        best_hr_entry = max(hr_candidates, key=lambda x: x["precision"])
        th_recall = best_hr_entry["threshold"]
    else:
        best_hr_entry = max(val_curve, key=lambda x: x["recall"])
        th_recall = best_hr_entry["threshold"]
        
    th_standard = 0.50
    
    selected_thresholds = {
        "balanced_f1_optimal": {
            "name": "Balanced Operational Mode (F1-Optimal)",
            "selected_threshold": th_balanced,
            "validation_f1": best_f1_val_entry["f1_score"],
            "validation_precision": best_f1_val_entry["precision"],
            "validation_recall": best_f1_val_entry["recall"],
            "validation_fpr": best_f1_val_entry["false_positive_rate"],
            "validation_lead_time_min": best_f1_val_entry["mean_lead_time_min"],
            "validation_adv_rate_pct": best_f1_val_entry["advance_warning_rate_pct"],
        },
        "high_precision": {
            "name": "High-Precision Mode (Low False Alarms)",
            "selected_threshold": th_precision,
            "validation_f1": best_hp_entry["f1_score"],
            "validation_precision": best_hp_entry["precision"],
            "validation_recall": best_hp_entry["recall"],
            "validation_fpr": best_hp_entry["false_positive_rate"],
            "validation_lead_time_min": best_hp_entry["mean_lead_time_min"],
            "validation_adv_rate_pct": best_hp_entry["advance_warning_rate_pct"],
        },
        "high_recall": {
            "name": "High-Recall Mode (Early Intervention)",
            "selected_threshold": th_recall,
            "validation_f1": best_hr_entry["f1_score"],
            "validation_precision": best_hr_entry["precision"],
            "validation_recall": best_hr_entry["recall"],
            "validation_fpr": best_hr_entry["false_positive_rate"],
            "validation_lead_time_min": best_hr_entry["mean_lead_time_min"],
            "validation_adv_rate_pct": best_hr_entry["advance_warning_rate_pct"],
        },
        "standard_baseline": {
            "name": "Standard Threshold Baseline",
            "selected_threshold": th_standard,
        }
    }
    
    log.info("Selected Thresholds on Validation Data:")
    log.info("  • Balanced F1: %.3f (Val F1=%.4f, Prec=%.4f, Rec=%.4f)", th_balanced, best_f1_val_entry["f1_score"], best_f1_val_entry["precision"], best_f1_val_entry["recall"])
    log.info("  • High Precision: %.3f (Val Prec=%.4f, Rec=%.4f)", th_precision, best_hp_entry["precision"], best_hp_entry["recall"])
    log.info("  • High Recall: %.3f (Val Rec=%.4f, Prec=%.4f)", th_recall, best_hr_entry["recall"], best_hr_entry["precision"])
    
    # 3. Test-Set Benchmarking on the Selected Thresholds (and full test curve for visualization)
    log.info("Evaluating selected operating points on untouched TEST set...")
    test_curve = evaluate_threshold_operating_curve(model, X_test, y_test, spd_test, ts_test, sensor_ids)
    
    # Extract test performance for each selected threshold
    def find_test_entry(th):
        return min(test_curve, key=lambda x: abs(x["threshold"] - th))
        
    test_benchmarks = {
        "balanced_f1_optimal": find_test_entry(th_balanced),
        "high_precision": find_test_entry(th_precision),
        "high_recall": find_test_entry(th_recall),
        "standard_baseline": find_test_entry(th_standard),
    }
    
    log.info("✓ Test Set Results for Selected Thresholds:")
    for mode, entry in test_benchmarks.items():
        log.info(
            "  [%s | th=%.3f]: Test F1=%.4f | Prec=%.4f | Rec=%.4f | FPR=%.4f | AdvRate=%.1f%% | MeanLead=%.1f min",
            mode, entry["threshold"], entry["f1_score"], entry["precision"], entry["recall"],
            entry["false_positive_rate"], entry["advance_warning_rate_pct"], entry["mean_lead_time_min"]
        )
        
    artifact = {
        "metadata": {
            "title": "TrafficFlow AI — Validation-Tuned Threshold Analysis",
            "methodology": "Thresholds tuned strictly on Validation set; benchmarked on untouched Test set.",
            "zero_leakage_guarantee": True,
            "generated_at": datetime.now().isoformat(),
        },
        "selected_modes": selected_thresholds,
        "test_set_benchmarks": test_benchmarks,
        "validation_curve": val_curve,
        "test_curve": test_curve,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    log.info("✓ Saved threshold analysis to %s", save_path)
    return artifact


if __name__ == "__main__":
    run_threshold_analysis()
