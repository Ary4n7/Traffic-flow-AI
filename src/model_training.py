"""
src/model_training.py
---------------------
STAGES 4, 5, 6, 7, 8 — Target Generation, Chronological Split, Model Training, & Comparison.

Trains:
1. Baseline Model: Temporal features only.
2. Proposed Model: Temporal + Spatial neighbor graph features.

Model Architecture:
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

Split:
Chronological 70% Train, 15% Val, 15% Test. (No data leakage or shuffling).
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids, load_distances
from src.preprocessing import clean_speed_matrix
from src.feature_engineering import extract_temporal_features, TEMPORAL_FEATURE_NAMES
from src.graph_builder import build_sensor_graph, compute_spatial_features, SPATIAL_FEATURE_NAMES
from src.evaluation import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("model_training")


def build_tabular_dataset(
    df_clean: pd.DataFrame,
    temp_feats: dict,
    sp_feats: dict,
    horizon_steps: int = 1, # 1 step = 5 min ahead
    congestion_threshold: float = config.CONGESTION_THRESHOLD_MPH,
    stride_step: int = 6, # 30-min stride: produces ~600k samples across 207 sensors for fast, representative training
) -> tuple:
    """
    Construct aligned tabular feature matrix X and binary target vector y across all sensors.
    """
    log.info("Constructing tabular dataset for horizon=%d steps (+%d min)...", horizon_steps, horizon_steps * 5)
    
    sensor_ids = list(df_clean.columns)
    timestamps = df_clean.index[3:-horizon_steps:stride_step] # skip warmup lags and horizon tail
    
    # Target matrix: speed at t + horizon_steps < threshold
    future_speeds = df_clean.shift(-horizon_steps)
    target_matrix = (future_speeds < congestion_threshold).astype(np.int8)
    
    rows_temp = []
    rows_spat = []
    y_list = []
    
    for t in timestamps:
        # Extract features for all sensors at timestamp t
        t_feats = np.column_stack([
            temp_feats[name].loc[t].values if isinstance(temp_feats[name], pd.DataFrame) 
            else np.full(len(sensor_ids), temp_feats[name].loc[t])
            for name in TEMPORAL_FEATURE_NAMES
        ])
        
        s_feats = np.column_stack([
            sp_feats[name].loc[t].values
            for name in SPATIAL_FEATURE_NAMES
        ])
        
        y_vals = target_matrix.loc[t].values
        
        rows_temp.append(t_feats)
        rows_spat.append(s_feats)
        y_list.append(y_vals)
        
    X_temp = np.vstack(rows_temp).astype(np.float32)
    X_spat = np.vstack(rows_spat).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int8)
    
    # Combined features for Proposed model
    X_combined = np.hstack([X_temp, X_spat])
    
    log.info("Dataset shape: X_temp=%s, X_combined=%s, y=%s (Congested rate: %.2f%%)", 
             X_temp.shape, X_combined.shape, y.shape, 100 * np.mean(y))
    return X_temp, X_combined, y, timestamps


def chronological_split(X: np.ndarray, y: np.ndarray, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split time-series observations chronologically."""
    N = len(y)
    n_train = int(N * train_frac)
    n_val = int(N * (train_frac + val_frac))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]
    
    log.info(
        "Chronological Split: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Test=%d (%.1f%%)",
        len(y_train), 100 * len(y_train) / N,
        len(y_val), 100 * len(y_val) / N,
        len(y_test), 100 * len(y_test) / N,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate_all():
    """Execute complete training, evaluation, and model artifact generation."""
    # 1. Load Data
    df_raw = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    
    # 2. Preprocess
    df_clean = clean_speed_matrix(df_raw)
    
    # 3. Features & Graph
    temp_feats = extract_temporal_features(df_clean)
    G = build_sensor_graph(dist_df, sensor_ids)
    sp_feats = compute_spatial_features(df_clean, G)
    
    # 4. Build Tabular Data (+5 min horizon default)
    X_temp, X_combined, y, _ = build_tabular_dataset(df_clean, temp_feats, sp_feats, horizon_steps=1)
    
    # 5. Chronological Split
    X_tr_base, y_train, X_va_base, y_val, X_te_base, y_test = chronological_split(X_temp, y)
    X_tr_prop, _, X_va_prop, _, X_te_prop, _ = chronological_split(X_combined, y)
    
    # 6. Train Baseline Model (Temporal Features Only)
    log.info("Training Baseline Model (Temporal features only)...")
    rf_baseline = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=22,
        min_samples_leaf=4,
        random_state=config.RF_RANDOM_STATE,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=config.RF_N_JOBS,
    )
    rf_baseline.fit(X_tr_base, y_train)
    
    # Evaluate Baseline
    y_pred_base = rf_baseline.predict(X_te_base)
    y_prob_base = rf_baseline.predict_proba(X_te_base)[:, 1]
    metrics_baseline = evaluate_classifier(y_test, y_pred_base, y_prob_base, model_name="Baseline (Temporal Only)")
    
    # 7. Train Proposed Model (Temporal + Spatial Graph Features)
    log.info("Training Proposed Model (Temporal + Spatial Graph Features)...")
    rf_spatial = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=22,
        min_samples_leaf=4,
        random_state=config.RF_RANDOM_STATE,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=config.RF_N_JOBS,
    )
    rf_spatial.fit(X_tr_prop, y_train)
    
    # Evaluate Proposed
    y_pred_prop = rf_spatial.predict(X_te_prop)
    y_prob_prop = rf_spatial.predict_proba(X_te_prop)[:, 1]
    metrics_spatial = evaluate_classifier(y_test, y_pred_prop, y_prob_prop, model_name="Proposed (Temporal + Spatial)")
    
    # 8. Feature Importances
    all_feature_names = TEMPORAL_FEATURE_NAMES + SPATIAL_FEATURE_NAMES
    importances = dict(zip(all_feature_names, [round(float(v), 4) for v in rf_spatial.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # 9. Save Artifacts
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.BASELINE_MODEL_PKL, "wb") as f:
        pickle.dump(rf_baseline, f)
    with open(config.SPATIAL_MODEL_PKL, "wb") as f:
        pickle.dump(rf_spatial, f)
        
    metadata = {
        "temporal_features": TEMPORAL_FEATURE_NAMES,
        "spatial_features": SPATIAL_FEATURE_NAMES,
        "all_features": all_feature_names,
        "feature_importances": sorted_importances,
        "congestion_threshold_mph": config.CONGESTION_THRESHOLD_MPH,
        "prediction_horizon_min": 5,
    }
    with open(config.FEATURE_META_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
        
    comparison_metrics = {
        "baseline": metrics_baseline,
        "proposed_spatial": metrics_spatial,
        "feature_importances": sorted_importances,
    }
    with open(config.METRICS_JSON, "w") as f:
        json.dump(comparison_metrics, f, indent=2)
        
    log.info("✓ All models trained and saved to %s", config.MODELS_DIR)
    return comparison_metrics


if __name__ == "__main__":
    train_and_evaluate_all()
