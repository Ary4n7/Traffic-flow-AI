"""
src/propagation_direction.py
----------------------------
STAGE 2 (Feature 3) — Data-Driven Probable Propagation Direction Engine.

Calculates time-lagged cross-correlations between graph-connected sensor pairs
to identify empirical lead-lag relationships in traffic flow without assuming
ground-truth road topology.

Key Principles:
1. Ground-Truth Honesty: Indian Driving Dataset (IDD) provides pairwise physical distances, NOT road-flow direction.
   We explicitly label inferred links as "Probable propagation direction" or
   "Graph neighbor — direction uncertain".
2. No Data Leakage: Cross-correlation is computed exclusively on the chronological
   training split (first 70% of the time series).
3. Efficiency: Only evaluates physically connected graph edges (from NetworkX adjacency)
   across 5, 10, and 15-minute lags.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_speed_data, load_sensor_ids, load_distances
from src.preprocessing import clean_speed_matrix
from src.graph_builder import build_sensor_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("propagation_direction")


def compute_lagged_cross_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    lag_steps: int = 1,
) -> float:
    """
    Compute Pearson cross-correlation between series_a(t - lag) and series_b(t).
    Positive correlation indicates changes in A precede corresponding changes in B.
    """
    # A leads B by lag_steps
    a_shifted = series_a.iloc[:-lag_steps].values
    b_target = series_b.iloc[lag_steps:].values
    
    # Mask any NaNs or constant segments
    valid = (~np.isnan(a_shifted)) & (~np.isnan(b_target))
    if np.sum(valid) < 50:
        return 0.0
        
    a_val = a_shifted[valid]
    b_val = b_target[valid]
    
    std_a = np.std(a_val)
    std_b = np.std(b_val)
    if std_a < 1e-4 or std_b < 1e-4:
        return 0.0
        
    corr = np.corrcoef(a_val, b_val)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def infer_probable_propagation_directions(
    df_train_speeds: pd.DataFrame,
    G: nx.Graph,
    lags_steps: list = config.PROPAGATION_LAGS_STEPS,
    corr_threshold: float = config.PROPAGATION_CORR_THRESH,
    conf_diff: float = config.PROPAGATION_CONF_DIFF,
    save_path: str = config.PROPAGATION_JSON,
) -> dict:
    """
    Evaluate lead-lag relationships for all connected graph edges in G using training speeds.
    
    Parameters
    ----------
    df_train_speeds : pd.DataFrame (T_train, N)
        Cleaned speed matrix for training split only.
    G : nx.Graph
        Undirected sensor proximity graph.
    lags_steps : list of int
        Time lags in 5-minute units (e.g. [1, 2, 3] -> 5m, 10m, 15m).
    corr_threshold : float
        Minimum correlation score required to establish a meaningful directional relationship.
    conf_diff : float
        Margin (r_fwd - r_rev) required to establish high/medium confidence direction.
    save_path : str
        Path to save the JSON artifact.
        
    Returns
    -------
    dict
        Full propagation dictionary with edge records, downstream mapping, and metadata.
    """
    log.info("Inferring probable propagation directions for %d graph edges...", G.number_of_edges())
    
    # Also compute 1-step delta speeds to capture dynamic deceleration shockwaves
    df_train_deltas = df_train_speeds.diff().fillna(0.0)
    
    edges_data = {}
    downstream_map = {node: [] for node in G.nodes()}
    upstream_map = {node: [] for node in G.nodes()}
    
    directional_count = 0
    uncertain_count = 0
    
    for u, v in G.edges():
        u, v = str(u), str(v)
        if u not in df_train_speeds.columns or v not in df_train_speeds.columns:
            continue
            
        s_u = df_train_speeds[u]
        s_v = df_train_speeds[v]
        d_u = df_train_deltas[u]
        d_v = df_train_deltas[v]
        
        # 1. Forward direction u -> v (u leads v)
        fwd_corrs = {}
        for lag in lags_steps:
            c_spd = compute_lagged_cross_correlation(s_u, s_v, lag_steps=lag)
            c_dlt = compute_lagged_cross_correlation(d_u, d_v, lag_steps=lag)
            # Blended correlation giving weight to both absolute speed level and speed deceleration
            fwd_corrs[lag * 5] = 0.65 * c_spd + 0.35 * max(0.0, c_dlt)
            
        # 2. Reverse direction v -> u (v leads u)
        rev_corrs = {}
        for lag in lags_steps:
            c_spd = compute_lagged_cross_correlation(s_v, s_u, lag_steps=lag)
            c_dlt = compute_lagged_cross_correlation(d_v, d_u, lag_steps=lag)
            rev_corrs[lag * 5] = 0.65 * c_spd + 0.35 * max(0.0, c_dlt)
            
        best_fwd_lag = max(fwd_corrs, key=fwd_corrs.get)
        best_fwd_score = fwd_corrs[best_fwd_lag]
        
        best_rev_lag = max(rev_corrs, key=rev_corrs.get)
        best_rev_score = rev_corrs[best_rev_lag]
        
        diff = best_fwd_score - best_rev_score
        
        # Classify directional influence
        if diff >= conf_diff and best_fwd_score >= corr_threshold:
            # u -> v is probable propagation direction
            conf = "High" if (diff >= 0.08 or best_fwd_score >= 0.65) else "Medium"
            dir_label = "Probable propagation direction"
            source, target = u, v
            best_lag = best_fwd_lag
            best_corr = best_fwd_score
            directional_count += 1
            
            entry = {
                "source_sensor": source,
                "target_sensor": target,
                "best_lag_minutes": int(best_lag),
                "correlation_score": round(float(best_corr), 4),
                "confidence": conf,
                "direction_label": dir_label,
                "lead_margin": round(float(diff), 4),
                "is_directional": True,
            }
            edges_data[f"{u}->{v}"] = entry
            downstream_map[u].append(entry)
            upstream_map[v].append(entry)
            
        elif diff <= -conf_diff and best_rev_score >= corr_threshold:
            # v -> u is probable propagation direction
            conf = "High" if (abs(diff) >= 0.08 or best_rev_score >= 0.65) else "Medium"
            dir_label = "Probable propagation direction"
            source, target = v, u
            best_lag = best_rev_lag
            best_corr = best_rev_score
            directional_count += 1
            
            entry = {
                "source_sensor": source,
                "target_sensor": target,
                "best_lag_minutes": int(best_lag),
                "correlation_score": round(float(best_corr), 4),
                "confidence": conf,
                "direction_label": dir_label,
                "lead_margin": round(float(abs(diff)), 4),
                "is_directional": True,
            }
            edges_data[f"{v}->{u}"] = entry
            downstream_map[v].append(entry)
            upstream_map[u].append(entry)
            
        else:
            # Symmetrical or weak evidence -> Direction Uncertain
            uncertain_count += 1
            entry = {
                "source_sensor": u,
                "target_sensor": v,
                "best_lag_minutes": int(best_fwd_lag),
                "correlation_score": round(float(max(best_fwd_score, best_rev_score)), 4),
                "confidence": "Low",
                "direction_label": "Graph neighbor — direction uncertain",
                "lead_margin": round(float(abs(diff)), 4),
                "is_directional": False,
            }
            edges_data[f"{u}<->{v}"] = entry
            
    # Sort downstream maps by correlation score
    for node in downstream_map:
        downstream_map[node].sort(key=lambda x: x["correlation_score"], reverse=True)
        
    artifact = {
        "metadata": {
            "title": "TrafficFlow AI — Probable Propagation Direction Mapping",
            "methodology": "Data-driven time-lagged cross-correlation on training set speed and delta series",
            "disclaimer": "Propagation direction is inferred probabilistically from temporal traffic relationships and should not be interpreted as ground-truth road direction.",
            "dataset": "Indian Driving Dataset (IDD)",
            "lags_evaluated_min": [int(l * 5) for l in lags_steps],
            "correlation_threshold": corr_threshold,
            "confidence_margin_threshold": conf_diff,
            "total_graph_edges": G.number_of_edges(),
            "inferred_directional_count": directional_count,
            "direction_uncertain_count": uncertain_count,
            "directional_ratio_pct": round(100.0 * directional_count / max(1, G.number_of_edges()), 1),
            "generated_at": datetime.now().isoformat(),
        },
        "edges": edges_data,
        "downstream_map": downstream_map,
        "upstream_map": upstream_map,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    log.info(
        "✓ Saved propagation directions to %s: %d directional (%0.1f%%), %d uncertain",
        save_path, directional_count, artifact["metadata"]["directional_ratio_pct"], uncertain_count
    )
    return artifact


class PropagationDirectionEngine:
    """Helper engine for querying inferred propagation relationships."""
    
    def __init__(self, artifact_path: str = config.PROPAGATION_JSON):
        if not os.path.exists(artifact_path):
            log.warning("Propagation artifact not found at %s. Running generation...", artifact_path)
            self.artifact = generate_and_save_propagation_directions(save_path=artifact_path)
        else:
            with open(artifact_path, "r") as f:
                self.artifact = json.load(f)
                
        self.edges = self.artifact.get("edges", {})
        self.downstream_map = self.artifact.get("downstream_map", {})
        self.upstream_map = self.artifact.get("upstream_map", {})
        self.metadata = self.artifact.get("metadata", {})

    def get_relationship(self, u: str, v: str) -> dict:
        """Get directional relationship between two sensors."""
        u, v = str(u), str(v)
        if f"{u}->{v}" in self.edges:
            return self.edges[f"{u}->{v}"]
        elif f"{v}->{u}" in self.edges:
            rev = self.edges[f"{v}->{u}"]
            return {
                "source_sensor": u,
                "target_sensor": v,
                "best_lag_minutes": rev["best_lag_minutes"],
                "correlation_score": rev["correlation_score"],
                "confidence": rev["confidence"],
                "direction_label": f"Probable upstream source (traffic flows {v} → {u})",
                "is_directional": True,
                "is_counter_flow": True,
            }
        elif f"{u}<->{v}" in self.edges:
            return self.edges[f"{u}<->{v}"]
        elif f"{v}<->{u}" in self.edges:
            return self.edges[f"{v}<->{u}"]
        else:
            return {
                "source_sensor": u,
                "target_sensor": v,
                "best_lag_minutes": 5,
                "correlation_score": 0.0,
                "confidence": "Low",
                "direction_label": "Graph neighbor — direction uncertain",
                "is_directional": False,
            }

    def get_downstream_candidates(self, u: str) -> list:
        """Return list of probable downstream candidate sensors for sensor u."""
        return self.downstream_map.get(str(u), [])


def generate_and_save_propagation_directions(
    save_path: str = config.PROPAGATION_JSON,
) -> dict:
    """Convenience pipeline to load training speeds, graph, and compute propagation directions."""
    df_raw = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    
    df_clean = clean_speed_matrix(df_raw)
    
    # Chronological training split
    n_train = int(len(df_clean) * config.TRAIN_FRAC)
    df_train = df_clean.iloc[:n_train]
    
    # Load or build graph
    if os.path.exists(config.ADJACENCY_PKL):
        with open(config.ADJACENCY_PKL, "rb") as f:
            G = pickle.load(f)
    else:
        G = build_sensor_graph(dist_df, sensor_ids)
        
    return infer_probable_propagation_directions(df_train, G, save_path=save_path)


if __name__ == "__main__":
    generate_and_save_propagation_directions()
