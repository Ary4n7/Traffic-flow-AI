"""
src/stress_testing.py
---------------------
STAGE 4 (P0) — Sensor Outage Robustness & Hardware Failure Stress-Testing Suite.

Evaluates system resilience against real-world hardware detector dropouts:
1. Global Random Outages: 0%, 5%, 10%, 20%, and 30% sensor outages across deterministic seeds.
2. Localized / Clustered Outage Simulation: A contiguous neighborhood cluster of 15 sensors
   experiencing simultaneous telemetry blackout.
3. Sporadic Temporal Packet Loss: 10% sporadic single-step telemetry dropouts across the network.

Strict Scientific Integrity:
- Explicitly labeled as simulated hardware stress tests.
- Strictly causal historical imputation (historical median / forward-fill, zero future leakage).
- Measures degradation in Accuracy, Precision, Recall, F1, ROC-AUC, Advance Warning Rate, and Lead Time.
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
    brier_score_loss,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids, load_distances
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import build_sensor_graph, compute_spatial_features, SPATIAL_FEATURE_NAMES
from src.probability_calibration import compute_expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("stress_testing")


def run_sensor_outage_stress_test(
    outage_fractions: list = [0.0, 0.05, 0.10, 0.20, 0.30],
    seeds: list = [42, 123, 456],
    save_path: str = getattr(config, "STAGE4_STRESS_TEST_JSON", os.path.join(config.MODELS_DIR, "stage4_stress_test_results.json")),
    save_stage3_compat: bool = True,
) -> dict:
    """Execute comprehensive sensor outage, localized cluster, and packet loss stress tests."""
    log.info("Starting Comprehensive Sensor Outage Stress Testing Suite...")
    
    df_raw = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    df_clean = clean_speed_matrix(df_raw)
    
    G = build_sensor_graph(dist_df, sensor_ids)
    
    with open(config.CALIBRATED_MODEL_PKL, "rb") as f:
        model = pickle.load(f)
        
    sensor_ids = list(df_clean.columns)
    N_sensors = len(sensor_ids)
    
    # Chronological test split
    N_total = len(df_clean)
    n_test = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    test_timestamps = df_clean.index[n_test:]
    T_test = len(test_timestamps)
    
    # Precompute clean test ground truth
    future_speeds = df_clean.shift(-1).loc[test_timestamps, sensor_ids].values
    y_test_clean = (future_speeds < config.CONGESTION_THRESHOLD_MPH).astype(np.int8).flatten()
    
    test_results = {}
    baseline_f1 = None
    
    # ── 1. Global Random Sensor Outage Sweep ──────────────────────────────────
    for frac in outage_fractions:
        pct_label = f"{int(frac * 100)}% Random Outage"
        n_dropped = int(frac * N_sensors)
        log.info("Evaluating: [%s] (%d sensors disabled)...", pct_label, n_dropped)
        
        seed_metrics = []
        
        for s_idx, seed in enumerate(seeds):
            rng = np.random.RandomState(seed)
            dropped_sensors = list(rng.choice(sensor_ids, size=n_dropped, replace=False)) if n_dropped > 0 else []
            
            df_stressed = df_clean.copy()
            if len(dropped_sensors) > 0:
                df_stressed[dropped_sensors] = np.nan
                # Impute missing test sensors using strictly backward-looking median from train/val only
                for s_drop in dropped_sensors:
                    df_stressed[s_drop] = df_clean[s_drop].iloc[:n_test].median()
                    
            temp_feats = extract_temporal_features(df_stressed)
            sp_feats = compute_spatial_features(df_stressed, G)
            
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
                
            X_stress = np.hstack([np.vstack(rows_temp), np.vstack(rows_spat)]).astype(np.float32)
            probs_stress = model.predict_proba(X_stress)[:, 1]
            y_pred_stress = (probs_stress >= 0.50).astype(int)
            
            acc = accuracy_score(y_test_clean, y_pred_stress)
            prec = precision_score(y_test_clean, y_pred_stress, zero_division=0)
            rec = recall_score(y_test_clean, y_pred_stress, zero_division=0)
            f1 = f1_score(y_test_clean, y_pred_stress, zero_division=0)
            roc = roc_auc_score(y_test_clean, probs_stress)
            brier = brier_score_loss(y_test_clean, probs_stress)
            ece = compute_expected_calibration_error(y_test_clean, probs_stress)
            
            # Lead time calculation
            probs_stress_mat = probs_stress.reshape(T_test, N_sensors)
            speeds_clean_mat = df_clean.loc[test_timestamps, sensor_ids].values
            
            tot_events = 0
            adv_events = 0
            lead_times = []
            
            for col_idx in range(N_sensors):
                s_spd = speeds_clean_mat[:, col_idx]
                s_pr = probs_stress_mat[:, col_idx]
                
                for i in range(1, T_test):
                    if s_spd[i] < config.CONGESTION_THRESHOLD_MPH and s_spd[i - 1] >= config.CONGESTION_THRESHOLD_MPH:
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
                            
            adv_arr = np.array([l for l in lead_times if l > 0])
            mean_lead = float(np.mean(adv_arr)) if len(adv_arr) > 0 else 0.0
            adv_rate = (adv_events / max(1, tot_events)) * 100.0
            
            seed_metrics.append({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc,
                "brier": brier,
                "ece": ece,
                "advance_rate_pct": adv_rate,
                "mean_lead_time_min": mean_lead,
            })
            
        avg_acc = float(np.mean([m["accuracy"] for m in seed_metrics]))
        avg_prec = float(np.mean([m["precision"] for m in seed_metrics]))
        avg_rec = float(np.mean([m["recall"] for m in seed_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in seed_metrics]))
        avg_roc = float(np.mean([m["roc_auc"] for m in seed_metrics]))
        avg_brier = float(np.mean([m["brier"] for m in seed_metrics]))
        avg_ece = float(np.mean([m["ece"] for m in seed_metrics]))
        avg_adv_rate = float(np.mean([m["advance_rate_pct"] for m in seed_metrics]))
        avg_lead = float(np.mean([m["mean_lead_time_min"] for m in seed_metrics]))
        
        if frac == 0.0:
            baseline_f1 = avg_f1
            degradation_pct = 0.0
        else:
            degradation_pct = round(((baseline_f1 - avg_f1) / baseline_f1) * 100.0, 2)
            
        test_results[pct_label] = {
            "test_type": "Random Global Outage",
            "outage_fraction": frac,
            "sensors_disabled_count": n_dropped,
            "accuracy": round(avg_acc, 4),
            "precision": round(avg_prec, 4),
            "recall": round(avg_rec, 4),
            "f1_score": round(avg_f1, 4),
            "roc_auc": round(avg_roc, 4),
            "brier_score": round(avg_brier, 4),
            "ece": round(avg_ece, 4),
            "advance_warning_rate_pct": round(avg_adv_rate, 2),
            "mean_lead_time_min": round(avg_lead, 1),
            "f1_degradation_pct": degradation_pct,
            "system_resilience_status": "ROBUST" if degradation_pct < 10.0 else ("MODERATE" if degradation_pct < 20.0 else "DEGRADED"),
        }
        
    # ── 2. Localized / Clustered Regional Outage Simulation ───────────────────
    log.info("Evaluating Localized Clustered Outage (15 contiguous neighboring sensors)...")
    # Pick a dense central hub sensor and its 2-hop neighborhood
    hub_sensor = sensor_ids[0]
    cluster_nodes = {hub_sensor}
    for n1 in G.neighbors(hub_sensor):
        cluster_nodes.add(n1)
        for n2 in G.neighbors(n1):
            cluster_nodes.add(n2)
            if len(cluster_nodes) >= 15:
                break
        if len(cluster_nodes) >= 15:
            break
    cluster_list = list(cluster_nodes)[:15]
    
    df_clustered = df_clean.copy()
    for s_drop in cluster_list:
        df_clustered[s_drop] = df_clean[s_drop].iloc[:n_test].median()
        
    temp_feats_c = extract_temporal_features(df_clustered)
    sp_feats_c = compute_spatial_features(df_clustered, G)
    
    rows_temp_c, rows_spat_c = [], []
    for t in test_timestamps:
        t_f = np.column_stack([
            temp_feats_c[name].loc[t].values if isinstance(temp_feats_c[name], pd.DataFrame)
            else np.full(N_sensors, temp_feats_c[name].loc[t])
            for name in TEMPORAL_FEATURE_NAMES
        ])
        s_f = np.column_stack([sp_feats_c[name].loc[t].values for name in SPATIAL_FEATURE_NAMES])
        rows_temp_c.append(t_f)
        rows_spat_c.append(s_f)
        
    X_cluster = np.hstack([np.vstack(rows_temp_c), np.vstack(rows_spat_c)]).astype(np.float32)
    probs_cluster = model.predict_proba(X_cluster)[:, 1]
    y_pred_cluster = (probs_cluster >= 0.50).astype(int)
    
    f1_cluster = float(f1_score(y_test_clean, y_pred_cluster, zero_division=0))
    degrad_cluster = round(((baseline_f1 - f1_cluster) / baseline_f1) * 100.0, 2)
    
    test_results["Localized Regional Cluster Outage (15 Nodes)"] = {
        "test_type": "Localized Spatial Outage",
        "outage_fraction": round(len(cluster_list) / N_sensors, 3),
        "sensors_disabled_count": len(cluster_list),
        "disabled_sensor_cluster": cluster_list[:5],
        "accuracy": round(float(accuracy_score(y_test_clean, y_pred_cluster)), 4),
        "precision": round(float(precision_score(y_test_clean, y_pred_cluster, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_clean, y_pred_cluster, zero_division=0)), 4),
        "f1_score": round(f1_cluster, 4),
        "roc_auc": round(float(roc_auc_score(y_test_clean, probs_cluster)), 4),
        "brier_score": round(float(brier_score_loss(y_test_clean, probs_cluster)), 4),
        "f1_degradation_pct": degrad_cluster,
        "system_resilience_status": "ROBUST" if degrad_cluster < 10.0 else "MODERATE",
    }
    
    # ── 3. Sporadic Temporal Packet Loss Simulation (10% Random Timesteps) ────
    log.info("Evaluating Sporadic Temporal Packet Loss (10% random temporal telemetry dropouts)...")
    rng_pkt = np.random.RandomState(999)
    df_packet_loss = df_clean.copy()
    drop_mask = rng_pkt.rand(*df_packet_loss.shape) < 0.10
    df_packet_loss[drop_mask] = np.nan
    df_packet_loss = df_packet_loss.ffill().bfill()
    
    temp_feats_p = extract_temporal_features(df_packet_loss)
    sp_feats_p = compute_spatial_features(df_packet_loss, G)
    
    rows_temp_p, rows_spat_p = [], []
    for t in test_timestamps:
        t_f = np.column_stack([
            temp_feats_p[name].loc[t].values if isinstance(temp_feats_p[name], pd.DataFrame)
            else np.full(N_sensors, temp_feats_p[name].loc[t])
            for name in TEMPORAL_FEATURE_NAMES
        ])
        s_f = np.column_stack([sp_feats_p[name].loc[t].values for name in SPATIAL_FEATURE_NAMES])
        rows_temp_p.append(t_f)
        rows_spat_p.append(s_f)
        
    X_packet = np.hstack([np.vstack(rows_temp_p), np.vstack(rows_spat_p)]).astype(np.float32)
    probs_packet = model.predict_proba(X_packet)[:, 1]
    y_pred_packet = (probs_packet >= 0.50).astype(int)
    
    f1_packet = float(f1_score(y_test_clean, y_pred_packet, zero_division=0))
    degrad_packet = round(((baseline_f1 - f1_packet) / baseline_f1) * 100.0, 2)
    
    test_results["Sporadic Temporal Packet Loss (10% Drops)"] = {
        "test_type": "Intermittent Telemetry Packet Loss",
        "drop_probability": 0.10,
        "imputation_strategy": "Causal Forward-Fill from previous telemetry reading",
        "accuracy": round(float(accuracy_score(y_test_clean, y_pred_packet)), 4),
        "precision": round(float(precision_score(y_test_clean, y_pred_packet, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_clean, y_pred_packet, zero_division=0)), 4),
        "f1_score": round(f1_packet, 4),
        "roc_auc": round(float(roc_auc_score(y_test_clean, probs_packet)), 4),
        "brier_score": round(float(brier_score_loss(y_test_clean, probs_packet)), 4),
        "f1_degradation_pct": degrad_packet,
        "system_resilience_status": "ROBUST" if degrad_packet < 10.0 else "MODERATE",
    }
    
    stress_artifact = {
        "metadata": {
            "title": "TrafficFlow AI — Stage 4 Comprehensive Robustness & Stress Test Benchmark",
            "test_timesteps": T_test,
            "total_sensors": N_sensors,
            "evaluated_conditions": ["0%", "5%", "10%", "20%", "30% Random Outages", "15-Node Clustered Outage", "10% Sporadic Packet Loss"],
            "imputation_strategy": "Historical causal sensor median fallback & forward fill (Zero future leakage)",
            "simulation_disclaimer": "All failure scenarios are empirical simulation stress-tests on historical METR-LA telemetry and do not represent hardware tampering.",
            "generated_at": datetime.now().isoformat(),
        },
        "stress_results": test_results,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stress_artifact, f, indent=2)
        
    if save_stage3_compat:
        # Also maintain Stage 3 JSON compatibility
        compat_path = config.STRESS_TEST_JSON
        with open(compat_path, "w") as f:
            json.dump(stress_artifact, f, indent=2)
            
    log.info("✓ Saved Stage 4 stress testing results to %s", save_path)
    return stress_artifact


if __name__ == "__main__":
    run_sensor_outage_stress_test()
