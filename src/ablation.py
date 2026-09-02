"""
src/ablation.py
---------------
STAGE 3 (Phase 6) — Comprehensive Ablation Study Benchmark Suite.

Evaluates progressive architectural configurations on the chronological TEST set:
- Model A: Temporal features only (11 features)
- Model B: Temporal + Spatial features (17 features)
- Model C: Temporal + Spatial + Propagation Direction integration
- Model D: Full Stage 3 System (Calibrated RF + TRM + SCP + PSI + Hysteresis)

Metrics evaluated: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR,
Brier Score, Advance Warning Rate, Mean Lead Time, Median Lead Time.
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
from src.data_loader import load_speed_data, load_sensor_ids
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import compute_spatial_features, SPATIAL_FEATURE_NAMES
from src.propagation_direction import PropagationDirectionEngine
from src.spatiotemporal_metrics import compute_all_spatiotemporal_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("ablation")


def run_ablation_study(
    save_path: str = os.path.join(config.MODELS_DIR, "stage3_ablation_benchmarks.json"),
) -> dict:
    """Execute complete ablation study across all 4 architectural configurations."""
    log.info("Starting Stage 3 Ablation Study...")
    
    df_raw = load_speed_data()
    df_clean = clean_speed_matrix(df_raw)
    
    with open(config.BASELINE_MODEL_PKL, "rb") as f:
        rf_baseline = pickle.load(f)
    with open(config.SPATIAL_MODEL_PKL, "rb") as f:
        rf_spatial = pickle.load(f)
        
    calibrated_model_path = os.path.join(config.MODELS_DIR, "calibrated_model.pkl")
    if os.path.exists(calibrated_model_path):
        with open(calibrated_model_path, "rb") as f:
            calibrated_model = pickle.load(f)
    else:
        calibrated_model = rf_spatial
        
    with open(config.ADJACENCY_PKL, "rb") as f:
        G = pickle.load(f)
        
    prop_engine = PropagationDirectionEngine()
    sensor_ids = list(df_clean.columns)
    N_sensors = len(sensor_ids)
    
    temp_feats = extract_temporal_features(df_clean)
    sp_feats = compute_spatial_features(df_clean, G)
    
    # Chronological Test Set (15%)
    N_total = len(df_clean)
    n_test = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    test_timestamps = df_clean.index[n_test:]
    T_test = len(test_timestamps)
    
    log.info("Building Test matrices across %d timestamps...", T_test)
    
    # 1. Feature matrices
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
        
    X_test_temp = np.vstack(rows_temp).astype(np.float32)
    X_test_spat = np.vstack(rows_spat).astype(np.float32)
    X_test_combined = np.hstack([X_test_temp, X_test_spat])
    
    # Ground truth: speed at t+1 < 20 mph
    future_speeds = df_clean.shift(-1).loc[test_timestamps, sensor_ids].values
    y_test = (future_speeds < config.CONGESTION_THRESHOLD_MPH).astype(np.int8).flatten()
    
    # Model A: Temporal Only
    log.info("Evaluating Model A (Temporal Only)...")
    probs_A = rf_baseline.predict_proba(X_test_temp)[:, 1]
    
    # Model B: Temporal + Spatial
    log.info("Evaluating Model B (Temporal + Spatial)...")
    probs_B = rf_spatial.predict_proba(X_test_combined)[:, 1]
    
    # Model C: Temporal + Spatial + Propagation Direction
    log.info("Evaluating Model C (Temporal + Spatial + Propagation)...")
    probs_B_mat = probs_B.reshape(T_test, N_sensors)
    speeds_mat = df_clean.loc[test_timestamps, sensor_ids].values
    
    probs_C_mat = np.zeros_like(probs_B_mat)
    probs_D_mat = np.zeros_like(probs_B_mat)
    
    # Calibrated base prob
    probs_cal_flat = calibrated_model.predict_proba(X_test_combined)[:, 1]
    probs_cal_mat = probs_cal_flat.reshape(T_test, N_sensors)
    
    for i, t in enumerate(test_timestamps):
        curr_spd_dict = {sensor_ids[c]: float(speeds_mat[i, c]) for c in range(N_sensors)}
        curr_prob_dict = {sensor_ids[c]: float(probs_B_mat[i, c]) for c in range(N_sensors)}
        prev_prob_dict = {sensor_ids[c]: float(probs_B_mat[i - 1, c]) for c in range(N_sensors)} if i > 0 else None
        
        # Propagation shock
        psi_dict = {}
        for s in sensor_ids:
            psi_dict[s] = prop_engine.get_downstream_candidates(s)
            
        metrics_dict = compute_all_spatiotemporal_metrics(
            sensor_ids=sensor_ids,
            graph=G,
            prop_engine=prop_engine,
            current_speeds=curr_spd_dict,
            current_rf_probs=curr_prob_dict,
            prev_rf_probs=prev_prob_dict,
        )
        
        for c, s in enumerate(sensor_ids):
            p_b = probs_B_mat[i, c]
            p_cal = probs_cal_mat[i, c]
            m = metrics_dict[s]
            
            # Model C: Adds propagation weight
            probs_C_mat[i, c] = np.clip(0.85 * p_b + 0.15 * m["PSI"], 0.0, 1.0)
            
            # Model D: Full Stage 3 Risk Fusion
            probs_D_mat[i, c] = np.clip(
                0.55 * p_cal + 0.20 * m["TRM"] + 0.15 * m["SCP"] + 0.10 * m["PSI"],
                0.0, 1.0
            )
            
    probs_C = probs_C_mat.flatten()
    probs_D = probs_D_mat.flatten()
    
    # Lead time calculator helper for any prob matrix
    def compute_lead_time_stats(probs_m):
        congestion_threshold = 20.0
        lead_times = []
        tot_events = 0
        adv_events = 0
        same_events = 0
        
        for col_idx in range(N_sensors):
            s_spd = speeds_mat[:, col_idx]
            s_pr = probs_m[:, col_idx]
            
            for i in range(1, T_test):
                if s_spd[i] < congestion_threshold and s_spd[i - 1] >= congestion_threshold:
                    tot_events += 1
                    k = i - 1
                    first_k = None
                    steps_b = 0
                    while k >= 0 and steps_b < 12:
                        if s_spd[k] < congestion_threshold:
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
        return {
            "total_events": tot_events,
            "advance_warned": adv_events,
            "advance_rate_pct": round(100.0 * adv_events / max(1, tot_events), 2),
            "total_detection_rate_pct": round(100.0 * (adv_events + same_events) / max(1, tot_events), 2),
            "mean_lead_time_min": round(float(np.mean(adv_arr)), 1) if len(adv_arr) > 0 else 0.0,
            "median_lead_time_min": round(float(np.median(adv_arr)), 1) if len(adv_arr) > 0 else 0.0,
        }
        
    log.info("Computing lead time statistics across models...")
    lt_A = compute_lead_time_stats(probs_A.reshape(T_test, N_sensors))
    lt_B = compute_lead_time_stats(probs_B_mat)
    lt_C = compute_lead_time_stats(probs_C_mat)
    lt_D = compute_lead_time_stats(probs_D_mat)
    
    def score_model(y_true, y_prob, name, lt_stats):
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            "architecture_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "brier_score": round(float(brier), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "confusion_matrix": cm.tolist(),
            "lead_time_stats": lt_stats,
        }
        
    ablation_results = {
        "metadata": {
            "title": "TrafficFlow AI — Stage 3 Ablation Benchmark Study",
            "test_timesteps": T_test,
            "total_test_samples": len(y_test),
            "congestion_threshold_mph": config.CONGESTION_THRESHOLD_MPH,
            "decision_threshold": 0.50,
            "generated_at": datetime.now().isoformat(),
        },
        "models": {
            "Model_A_Temporal_Only": score_model(y_test, probs_A, "Model A (Temporal Only)", lt_A),
            "Model_B_Temporal_Spatial": score_model(y_test, probs_B, "Model B (Temporal + Spatial)", lt_B),
            "Model_C_Temporal_Spatial_Propagation": score_model(y_test, probs_C, "Model C (Temporal + Spatial + Propagation)", lt_C),
            "Model_D_Full_Stage_3_System": score_model(y_test, probs_D, "Model D (Full Stage 3: Calibrated + TRM + SCP + PSI)", lt_D),
        },
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
        
    log.info("✓ Saved ablation study results to %s", save_path)
    return ablation_results


if __name__ == "__main__":
    run_ablation_study()
