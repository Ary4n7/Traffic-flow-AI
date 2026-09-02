"""
src/graph_builder.py
--------------------
STAGE 3 — Graph Construction + Spatial Feature Extraction.

Constructs the road-network sensor proximity graph from METR-LA distances
and computes spatial features for every sensor using NetworkX.

Features calculated per sensor:
- neighbor_mean_speed
- neighbor_min_speed
- neighbor_max_speed
- neighbor_congested_count
- neighbor_congestion_ratio
- neighbor_speed_delta (target speed - neighbor_mean_speed)
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("graph_builder")


SPATIAL_FEATURE_NAMES = [
    "neighbor_mean_speed",
    "neighbor_min_speed",
    "neighbor_max_speed",
    "neighbor_congested_count",
    "neighbor_congestion_ratio",
    "neighbor_speed_delta",
]


def build_sensor_graph(
    dist_df: pd.DataFrame,
    sensor_ids: list,
    sigma: float = 3500.0, # distance std in meters (~3.5km for local road adjacency)
    min_weight: float = 0.15,
    max_k_neighbors: int = 8,
    save_path: str = config.ADJACENCY_PKL,
) -> nx.Graph:
    """
    Construct an undirected NetworkX sensor proximity graph from pairwise distances.
    
    Using Gaussian thresholded distance kernel with localized neighbor filtering:
        W_ij = exp( - (dist_ij / sigma)^2 )
    """
    log.info("Building NetworkX proximity graph for %d sensors...", len(sensor_ids))
    G = nx.Graph()
    
    # Add all 207 sensors as nodes
    for s_id in sensor_ids:
        G.add_node(str(s_id))
        
    sensor_set = set(sensor_ids)
    
    # Filter distances to only include sensors present in our dataset
    df_edges = dist_df[
        (dist_df["from"].isin(sensor_set)) & 
        (dist_df["to"].isin(sensor_set)) & 
        (dist_df["from"] != dist_df["to"])
    ].copy()
    
    # Compute Gaussian kernel weights
    df_edges["weight"] = np.exp(-np.square(df_edges["cost"] / sigma))
    
    # Filter by minimum weight threshold (keep meaningful close neighbors)
    df_valid = df_edges[df_edges["weight"] >= min_weight]
    
    # For each sensor, connect to up to max_k_neighbors closest sensors
    for s_id in sensor_ids:
        s_edges = df_valid[df_valid["from"] == str(s_id)].sort_values("cost").head(max_k_neighbors)
        if len(s_edges) == 0:
            # Fallback to nearest 2 neighbors even if below weight threshold
            s_edges = df_edges[df_edges["from"] == str(s_id)].sort_values("cost").head(2)
            
        for _, row in s_edges.iterrows():
            u, v, cost, weight = str(row["from"]), str(row["to"]), float(row["cost"]), float(row["weight"])
            G.add_edge(u, v, distance=cost, weight=weight)
                
    log.info(
        "Sensor graph built: %d nodes, %d edges  (avg degree: %.2f)",
        G.number_of_nodes(),
        G.number_of_edges(),
        np.mean([d for _, d in G.degree()]),
    )
    
    # Save graph artifact
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(G, f)
    log.info("Saved graph to %s", save_path)
    
    return G


def compute_spatial_features(
    df_speeds: pd.DataFrame,
    G: nx.Graph,
    congestion_threshold: float = config.CONGESTION_THRESHOLD_MPH,
) -> dict:
    """
    Compute spatial neighbor features across all sensors and timestamps.
    
    Returns
    -------
    dict of str -> pd.DataFrame (T, N)
        Spatial feature matrices aligned with df_speeds.
    """
    log.info("Computing spatial features for %d sensors...", len(df_speeds.columns))
    
    sensor_ids = list(df_speeds.columns)
    T = len(df_speeds)
    
    neighbor_mean = pd.DataFrame(index=df_speeds.index, columns=sensor_ids, dtype="float32")
    neighbor_min = pd.DataFrame(index=df_speeds.index, columns=sensor_ids, dtype="float32")
    neighbor_max = pd.DataFrame(index=df_speeds.index, columns=sensor_ids, dtype="float32")
    neighbor_congested_count = pd.DataFrame(index=df_speeds.index, columns=sensor_ids, dtype="float32")
    neighbor_congestion_ratio = pd.DataFrame(index=df_speeds.index, columns=sensor_ids, dtype="float32")
    
    # Binary congestion indicator matrix for fast lookup
    is_congested_matrix = (df_speeds < congestion_threshold).astype(np.float32)
    
    for s_id in sensor_ids:
        neighbors = list(G.neighbors(str(s_id)))
        if not neighbors:
            neighbors = [s_id]
            
        neigh_speeds = df_speeds[neighbors]
        neigh_congested = is_congested_matrix[neighbors]
        
        neighbor_mean[s_id] = neigh_speeds.mean(axis=1)
        neighbor_min[s_id] = neigh_speeds.min(axis=1)
        neighbor_max[s_id] = neigh_speeds.max(axis=1)
        neighbor_congested_count[s_id] = neigh_congested.sum(axis=1)
        neighbor_congestion_ratio[s_id] = neigh_congested.mean(axis=1)
        
    neighbor_speed_delta = df_speeds - neighbor_mean
    
    spatial_features = {
        "neighbor_mean_speed": neighbor_mean,
        "neighbor_min_speed": neighbor_min,
        "neighbor_max_speed": neighbor_max,
        "neighbor_congested_count": neighbor_congested_count,
        "neighbor_congestion_ratio": neighbor_congestion_ratio,
        "neighbor_speed_delta": neighbor_speed_delta,
    }
    
    log.info("✓ Spatial features computed successfully (%d feature matrices)", len(spatial_features))
    return spatial_features


if __name__ == "__main__":
    from src.data_loader import load_speed_data, load_sensor_ids, load_distances
    from src.preprocessing import clean_speed_matrix
    
    df_raw = load_speed_data()
    df_clean = clean_speed_matrix(df_raw)
    sensor_ids = load_sensor_ids()
    dist_df = load_distances()
    
    G = build_sensor_graph(dist_df, sensor_ids)
    sp_feats = compute_spatial_features(df_clean, G)
    
    print("\nSample spatial features for sensor", sensor_ids[0])
    s_sample = pd.DataFrame({k: sp_feats[k][sensor_ids[0]] for k in sp_feats}).head()
    print(s_sample)
