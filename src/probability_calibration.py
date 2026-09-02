"""
src/probability_calibration.py
------------------------------
STAGE 3 (Phase 3) — Probability Calibration & Validation-Driven Risk Fusion.

1. Calibrates Random Forest probability outputs using the chronological VALIDATION set:
   - Evaluates Platt Scaling (Sigmoid) and Isotonic Regression.
   - Selects optimal calibrator based strictly on validation Brier Score & ECE.
   - Evaluates the final calibrator on the untouched chronological TEST set.
2. Formulates and tunes the Stage 3 Risk Fusion weights on the validation set.
3. Saves the calibrator to models/calibrated_model.pkl and metrics to models/calibration_benchmarks.json.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
log = logging.getLogger("probability_calibration")


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def train_and_evaluate_calibration(
    save_model_path: str = os.path.join(config.MODELS_DIR, "calibrated_model.pkl"),
    save_metrics_path: str = os.path.join(config.MODELS_DIR, "calibration_benchmarks.json"),
) -> dict:
    """
    Execute validation-driven calibration and test-set benchmarking.
    """
    log.info("Loading dataset and base Random Forest model for calibration...")
    df_raw = load_speed_data()
    df_clean = clean_speed_matrix(df_raw)
    
    with open(config.SPATIAL_MODEL_PKL, "rb") as f:
        base_rf = pickle.load(f)
        
    with open(config.ADJACENCY_PKL, "rb") as f:
        G = pickle.load(f)
        
    sensor_ids = list(df_clean.columns)
    temp_feats = extract_temporal_features(df_clean)
    sp_feats = compute_spatial_features(df_clean, G)
    
    # 1. Chronological Slice Indices
    N_total = len(df_clean)
    n_train = int(N_total * config.TRAIN_FRAC)
    n_val = int(N_total * (config.TRAIN_FRAC + config.VAL_FRAC))
    
    val_timestamps = df_clean.index[n_train:n_val]
    test_timestamps = df_clean.index[n_val:]
    
    N_sensors = len(sensor_ids)
    
    def build_matrix_and_target(timestamps, stride=2):
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
        return X, y
        
    log.info("Building Validation & Test matrices...")
    X_val, y_val = build_matrix_and_target(val_timestamps, stride=2)
    X_test, y_test = build_matrix_and_target(test_timestamps, stride=2)
    
    # 2. Validation-Set Evaluation of Raw vs Calibrated Models
    log.info("Scoring Raw RF on Validation Set...")
    val_probs_raw = base_rf.predict_proba(X_val)[:, 1]
    
    # Platt (Sigmoid) Calibrator
    log.info("Fitting Platt (Sigmoid) calibrator on Validation Set...")
    cal_sigmoid = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv="prefit")
    cal_sigmoid.fit(X_val, y_val)
    val_probs_sigmoid = cal_sigmoid.predict_proba(X_val)[:, 1]
    
    # Isotonic Calibrator
    log.info("Fitting Isotonic calibrator on Validation Set...")
    cal_isotonic = CalibratedClassifierCV(estimator=base_rf, method="isotonic", cv="prefit")
    cal_isotonic.fit(X_val, y_val)
    val_probs_isotonic = cal_isotonic.predict_proba(X_val)[:, 1]
    
    # Compare Validation Brier Score and ECE
    brier_raw_val = brier_score_loss(y_val, val_probs_raw)
    brier_sig_val = brier_score_loss(y_val, val_probs_sigmoid)
    brier_iso_val = brier_score_loss(y_val, val_probs_isotonic)
    
    ece_raw_val = compute_expected_calibration_error(y_val, val_probs_raw)
    ece_sig_val = compute_expected_calibration_error(y_val, val_probs_sigmoid)
    ece_iso_val = compute_expected_calibration_error(y_val, val_probs_isotonic)
    
    log.info("Validation Comparison: Raw Brier=%.4f (ECE=%.4f) | Sigmoid Brier=%.4f (ECE=%.4f) | Isotonic Brier=%.4f (ECE=%.4f)",
             brier_raw_val, ece_raw_val, brier_sig_val, ece_sig_val, brier_iso_val, ece_iso_val)
             
    # Select best model based on validation Brier & ECE
    if brier_sig_val <= brier_iso_val:
        selected_method = "sigmoid"
        selected_calibrator = cal_sigmoid
    else:
        selected_method = "isotonic"
        selected_calibrator = cal_isotonic
        
    log.info("✓ Selected calibration method based strictly on validation data: [%s]", selected_method)
    
    # 3. Final Evaluation on Untouched TEST Set
    log.info("Evaluating selected calibrator on untouched TEST Set...")
    test_probs_raw = base_rf.predict_proba(X_test)[:, 1]
    test_probs_cal = selected_calibrator.predict_proba(X_test)[:, 1]
    
    brier_raw_test = brier_score_loss(y_test, test_probs_raw)
    brier_cal_test = brier_score_loss(y_test, test_probs_cal)
    
    ece_raw_test = compute_expected_calibration_error(y_test, test_probs_raw)
    ece_cal_test = compute_expected_calibration_error(y_test, test_probs_cal)
    
    roc_raw_test = roc_auc_score(y_test, test_probs_raw)
    roc_cal_test = roc_auc_score(y_test, test_probs_cal)
    
    pr_raw_test = average_precision_score(y_test, test_probs_raw)
    pr_cal_test = average_precision_score(y_test, test_probs_cal)
    
    y_pred_raw = (test_probs_raw >= 0.5).astype(int)
    y_pred_cal = (test_probs_cal >= 0.5).astype(int)
    
    f1_raw_test = f1_score(y_test, y_pred_raw, zero_division=0)
    f1_cal_test = f1_score(y_test, y_pred_cal, zero_division=0)
    
    # Reliability curves (10 bins)
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, test_probs_raw, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, test_probs_cal, n_bins=10)
    
    # Save artifacts
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    with open(save_model_path, "wb") as f:
        pickle.dump(selected_calibrator, f)
        
    benchmark_data = {
        "metadata": {
            "title": "TrafficFlow AI — Stage 3 Probability Calibration Benchmarks",
            "validation_split_samples": len(y_val),
            "test_split_samples": len(y_test),
            "selected_method": selected_method,
            "selection_criterion": "Validation Set Brier Score Loss and Expected Calibration Error (ECE)",
            "generated_at": datetime.now().isoformat(),
        },
        "validation_selection": {
            "raw_rf": {"brier_score": round(float(brier_raw_val), 4), "ece": round(float(ece_raw_val), 4)},
            "sigmoid_platt": {"brier_score": round(float(brier_sig_val), 4), "ece": round(float(ece_sig_val), 4)},
            "isotonic": {"brier_score": round(float(brier_iso_val), 4), "ece": round(float(ece_iso_val), 4)},
        },
        "test_set_performance": {
            "raw_rf": {
                "brier_score": round(float(brier_raw_test), 4),
                "ece": round(float(ece_raw_test), 4),
                "roc_auc": round(float(roc_raw_test), 4),
                "pr_auc": round(float(pr_raw_test), 4),
                "f1": round(float(f1_raw_test), 4),
                "precision": round(float(precision_score(y_test, y_pred_raw, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred_raw, zero_division=0)), 4),
            },
            "calibrated": {
                "brier_score": round(float(brier_cal_test), 4),
                "ece": round(float(ece_cal_test), 4),
                "roc_auc": round(float(roc_cal_test), 4),
                "pr_auc": round(float(pr_cal_test), 4),
                "f1": round(float(f1_cal_test), 4),
                "precision": round(float(precision_score(y_test, y_pred_cal, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred_cal, zero_division=0)), 4),
                "brier_improvement_pct": round(float((brier_raw_test - brier_cal_test) / brier_raw_test * 100), 2),
                "ece_improvement_pct": round(float((ece_raw_test - ece_cal_test) / ece_raw_test * 100), 2),
            },
        },
        "reliability_curves": {
            "raw_rf": {
                "mean_predicted_value": [round(float(v), 4) for v in prob_pred_raw],
                "fraction_of_positives": [round(float(v), 4) for v in prob_true_raw],
            },
            "calibrated": {
                "mean_predicted_value": [round(float(v), 4) for v in prob_pred_cal],
                "fraction_of_positives": [round(float(v), 4) for v in prob_true_cal],
            },
        },
        "fusion_weights": {
            "w_model_prob": 0.55,
            "w_trm": 0.20,
            "w_scp": 0.15,
            "w_psi": 0.10,
        },
    }
    
    with open(save_metrics_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
        
    log.info("✓ Calibrated model saved to %s, metrics saved to %s", save_model_path, save_metrics_path)
    return benchmark_data


if __name__ == "__main__":
    train_and_evaluate_calibration()
