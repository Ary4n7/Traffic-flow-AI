"""
config.py
---------
Central configuration for TrafficFlow AI.

All thresholds, paths, and hyperparameters live here so they can be
changed in one place without touching source files.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR        = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR  = os.path.join(ROOT_DIR, "data", "processed")
DATA_SIM_DIR        = os.path.join(ROOT_DIR, "data", "simulation")
MODELS_DIR          = os.path.join(ROOT_DIR, "models")

METR_LA_H5          = os.path.join(DATA_RAW_DIR, "Indian Driving Dataset (IDD).h5")
DISTANCES_CSV       = os.path.join(DATA_RAW_DIR, "distances_delhi_ncr.csv")
SENSOR_IDS_TXT      = os.path.join(DATA_RAW_DIR, "graph_sensor_ids.txt")

PROCESSED_SPEEDS    = os.path.join(DATA_PROCESSED_DIR, "speeds.parquet")
ADJACENCY_PKL       = os.path.join(DATA_PROCESSED_DIR, "adjacency.pkl")

BASELINE_MODEL_PKL  = os.path.join(MODELS_DIR, "rf_baseline.pkl")
SPATIAL_MODEL_PKL   = os.path.join(MODELS_DIR, "rf_spatial.pkl")
FEATURE_META_JSON   = os.path.join(MODELS_DIR, "feature_metadata.json")
METRICS_JSON        = os.path.join(MODELS_DIR, "evaluation_metrics.json")
PROPAGATION_JSON    = os.path.join(MODELS_DIR, "propagation_direction.json")
LEAD_TIME_METRICS_JSON = os.path.join(MODELS_DIR, "warning_lead_time_metrics.json")

# ── Dataset ──────────────────────────────────────────────────────────────────
# The Indian Driving Dataset (IDD) dataset records speed in miles per hour (mph).
SPEED_UNIT          = "mph"
SAMPLE_INTERVAL_MIN = 5          # minutes between each reading

# ── Congestion definition ────────────────────────────────────────────────────
# A sensor reading below this speed is labelled as "congested".
# The default is 20 mph, which is a reasonable operational choice for this
# dataset.  It is configurable — change it here to experiment.
CONGESTION_THRESHOLD_MPH = 20    # mph

# ── Prediction horizons ──────────────────────────────────────────────────────
# How many timesteps ahead to predict.  One timestep = SAMPLE_INTERVAL_MIN.
PREDICTION_HORIZONS = {
    "5min":  1,   # 1 step  → 5 minutes
    "10min": 2,   # 2 steps → 10 minutes
    "15min": 3,   # 3 steps → 15 minutes
}
DEFAULT_HORIZON = "5min"    # horizon used when training a single model

# ── Graph construction ───────────────────────────────────────────────────────
# Gaussian kernel parameters for building the adjacency matrix from distances.
# Edges with weight < ADJ_WEIGHT_THRESHOLD are dropped (sparse graph).
ADJ_SIGMA_SQUARED   = 10.0   # σ²  for exp(−d²/σ²)
ADJ_WEIGHT_THRESHOLD = 0.1   # minimum edge weight to keep

# Maximum number of graph hops for spatial features (1 = immediate neighbors).
SPATIAL_HOPS        = 1

# ── Temporal feature lags ─────────────────────────────────────────────────────
# Lags in timesteps (each timestep = 5 min).
SPEED_LAGS_STEPS    = [1, 2, 3]   # → t-5min, t-10min, t-15min
ROLLING_WINDOW_STEPS = 6           # 6 × 5min = 30-min rolling window

# ── Train / validation / test split ──────────────────────────────────────────
# Chronological split — never shuffle time-series data.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC is the remainder (0.15)

# ── Random Forest hyperparameters ────────────────────────────────────────────
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = None       # None = grow full trees (controlled by min samples)
RF_RANDOM_STATE  = 42
RF_CLASS_WEIGHT  = "balanced"
RF_N_JOBS        = -1         # use all available CPU cores

# ── Risk thresholds (for dashboard display) ───────────────────────────────────
RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.00),
}

# ── Early Warning States (Stage 2) ────────────────────────────────────────────
# 5-tier operational alert hierarchy combining ML probability and temporal speed decay
EARLY_WARNING_STATES = {
    "NORMAL": {
        "label": "NORMAL",
        "description": "Stable highway flow and low predicted congestion risk (<30%)",
        "color": "#22C55E",
        "badge_color": "green",
    },
    "WATCH": {
        "label": "WATCH",
        "description": "Early speed deterioration or moderate risk (30–50%)",
        "color": "#EAB308",
        "badge_color": "yellow",
    },
    "WARNING": {
        "label": "WARNING",
        "description": "Significant deterioration or high risk (50–70%)",
        "color": "#F59E0B",
        "badge_color": "orange",
    },
    "HIGH RISK": {
        "label": "HIGH RISK",
        "description": "Imminent congestion build-up (70–85% risk)",
        "color": "#F97316",
        "badge_color": "red",
    },
    "CRITICAL": {
        "label": "CRITICAL",
        "description": "Currently congested (<20 mph) or extreme risk (≥85%)",
        "color": "#EF4444",
        "badge_color": "red",
    },
}

# ── Probable Propagation Direction Inference (Stage 2) ────────────────────────
# Lead-lag cross-correlation evaluation on graph-connected sensor pairs
PROPAGATION_LAGS_STEPS    = [1, 2, 3]    # 5, 10, 15 minutes lag
PROPAGATION_CORR_THRESH   = 0.30         # minimum correlation score to establish directional link
PROPAGATION_CONF_DIFF     = 0.04         # minimum asymmetry margin (r_fwd - r_rev) for High/Med confidence

# ── Stage 3: Alert Hysteresis & Debouncing Parameters ─────────────────────────
HYSTERESIS_ENABLED = True
HYSTERESIS_UPGRADE_PATIENCE = 1       # Fast emergency escalation (1 step = 5 min)
HYSTERESIS_DOWNGRADE_PATIENCE = 2     # Sustained recovery required to de-escalate (2 steps = 10 min)
HYSTERESIS_PROB_DEADBAND = 0.03       # Threshold deadband margin to suppress boundary jitter

# ── Stage 3: Spatiotemporal Risk Fusion Weights (Validation Optimized) ────────
STAGE3_FUSION_WEIGHTS = {
    "w_model_prob": 0.55,    # Calibrated ML risk probability
    "w_trm":        0.20,    # Temporal Risk Momentum
    "w_scp":        0.15,    # Spatial Congestion Pressure
    "w_psi":        0.10,    # Propagation Shock-Wave Index
}

# ── Stage 3 Artifact Paths ───────────────────────────────────────────────────
CALIBRATED_MODEL_PKL       = os.path.join(MODELS_DIR, "calibrated_model.pkl")
CALIBRATION_JSON           = os.path.join(MODELS_DIR, "calibration_benchmarks.json")
ABLATION_JSON              = os.path.join(MODELS_DIR, "stage3_ablation_benchmarks.json")
STRESS_TEST_JSON           = os.path.join(MODELS_DIR, "stage3_stress_test_results.json")

# ── Stage 4: Multi-Horizon & Robustness Artifact Paths ─────────────────────────
SPATIAL_MODEL_10MIN_PKL    = os.path.join(MODELS_DIR, "rf_spatial_10min.pkl")
SPATIAL_MODEL_15MIN_PKL    = os.path.join(MODELS_DIR, "rf_spatial_15min.pkl")
CALIBRATED_MODEL_10MIN_PKL = os.path.join(MODELS_DIR, "calibrated_10min.pkl")
CALIBRATED_MODEL_15MIN_PKL = os.path.join(MODELS_DIR, "calibrated_15min.pkl")
MULTI_HORIZON_METRICS_JSON = os.path.join(MODELS_DIR, "multi_horizon_metrics.json")
THRESHOLD_ANALYSIS_JSON    = os.path.join(MODELS_DIR, "threshold_analysis.json")
TEMPORAL_REGIME_JSON       = os.path.join(MODELS_DIR, "temporal_regime_benchmarks.json")
PROPAGATION_VALIDATION_JSON = os.path.join(MODELS_DIR, "propagation_validation.json")
STAGE4_STRESS_TEST_JSON    = os.path.join(MODELS_DIR, "stage4_stress_test_results.json")

# ── Simulator ────────────────────────────────────────────────────────────────
SIM_MORNING_RUSH = {"start": "07:00", "end": "10:00", "intensity": 0.75}
SIM_EVENING_RUSH = {"start": "16:00", "end": "20:00", "intensity": 0.85}
SIM_INCIDENT_PROB = 0.02
