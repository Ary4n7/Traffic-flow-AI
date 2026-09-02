"""
src/spatiotemporal_metrics.py
-----------------------------
STAGE 3 (Phase 1) — Mathematical Spatiotemporal Traffic Risk Metrics.

Formulates and computes three standardized, zero-leakage composite indices:
1. Temporal Risk Momentum (TRM): Rate of risk acceleration and deceleration velocity.
2. Spatial Congestion Pressure (SCP): Localized graph neighborhood congestion stress.
3. Propagation Shock-Wave Index (PSI): Direction-aware upstream/downstream wave pressure.

All metrics are strictly normalized to [0.0, 1.0], deterministic, and use ONLY
observations up to timestamp t.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional


def compute_temporal_risk_momentum(
    current_prob: float,
    prev_prob: Optional[float] = None,
    current_speed: float = 55.0,
    speed_delta_5: float = 0.0,
    speed_delta_10: float = 0.0,
    rolling_trend_30: float = 0.0,
    ref_speed_drop: float = 25.0,
) -> float:
    """
    Compute Temporal Risk Momentum (TRM) for a single sensor.
    
    Mathematical Definition:
        TRM(t) = clip( 0.40 * P(t)
                     + 0.25 * max(0, ΔP(t)) / 0.5
                     + 0.20 * max(0, -Δ_10(s)) / ref_drop
                     + 0.15 * max(0, -trend_30(s)) / ref_drop, 0.0, 1.0 )
    """
    p_curr = float(np.clip(current_prob, 0.0, 1.0))
    delta_p = (p_curr - float(prev_prob)) if prev_prob is not None else 0.0
    p_velocity = max(0.0, delta_p) / 0.50 # Normalized velocity
    
    # Deceleration components (positive when traffic is slowing down)
    decel_10 = max(0.0, -float(speed_delta_10)) / ref_speed_drop
    trend_decay = max(0.0, -float(rolling_trend_30)) / ref_speed_drop
    
    raw_trm = (
        0.40 * p_curr +
        0.25 * np.clip(p_velocity, 0.0, 1.0) +
        0.20 * np.clip(decel_10, 0.0, 1.0) +
        0.15 * np.clip(trend_decay, 0.0, 1.0)
    )
    return float(np.clip(raw_trm, 0.0, 1.0))


def compute_spatial_congestion_pressure(
    sensor_id: str,
    graph: nx.Graph,
    current_speeds: Dict[str, float],
    current_probs: Dict[str, float],
    congestion_threshold: float = 20.0,
    free_flow_speed: float = 65.0,
) -> float:
    """
    Compute Spatial Congestion Pressure (SCP) for a single sensor.
    
    Mathematical Definition:
        SCP(u, t) = clip( 0.45 * NeighborCongestionRatio(u)
                        + 0.35 * MeanNeighborRisk(u)
                        + 0.20 * max(0, v_free - MeanNeighborSpeed(u)) / v_free, 0.0, 1.0 )
    """
    sensor_id = str(sensor_id)
    neighbors = list(graph.neighbors(sensor_id)) if sensor_id in graph else []
    if not neighbors:
        return 0.0
        
    n_speeds = [float(current_speeds.get(n, 55.0)) for n in neighbors]
    n_probs = [float(current_probs.get(n, 0.0)) for n in neighbors]
    
    # 1. Congestion ratio
    cong_count = sum(1 for s in n_speeds if s < congestion_threshold)
    cong_ratio = cong_count / len(neighbors)
    
    # 2. Mean neighbor risk
    mean_risk = float(np.mean(n_probs))
    
    # 3. Neighbor speed deficit
    mean_spd = float(np.mean(n_speeds))
    spd_deficit = max(0.0, free_flow_speed - mean_spd) / free_flow_speed
    
    raw_scp = (
        0.45 * cong_ratio +
        0.35 * mean_risk +
        0.20 * np.clip(spd_deficit, 0.0, 1.0)
    )
    return float(np.clip(raw_scp, 0.0, 1.0))


def compute_propagation_shock_index(
    sensor_id: str,
    graph: nx.Graph,
    prop_engine,
    current_speeds: Dict[str, float],
    current_probs: Dict[str, float],
    prev_probs: Optional[Dict[str, float]] = None,
    congestion_threshold: float = 20.0,
) -> float:
    """
    Compute Propagation Shock-Wave Index (PSI) for a single sensor.
    
    Mathematical Definition:
        PSI(u, t) = clip( 0.60 * UpstreamFeederPressure(u, t)
                        + 0.40 * DownstreamBackpressure(u, t), 0.0, 1.0 )
    """
    sensor_id = str(sensor_id)
    if not prop_engine or sensor_id not in graph:
        return 0.0
        
    curr_spd = float(current_speeds.get(sensor_id, 55.0))
    neighbors = list(graph.neighbors(sensor_id))
    if not neighbors:
        return 0.0
        
    upstream_scores = []
    downstream_scores = []
    
    for n in neighbors:
        rel = prop_engine.get_relationship(n, sensor_id) # n -> sensor_id
        conf = rel.get("confidence", "Low")
        conf_weight = 1.0 if conf == "High" else (0.65 if conf == "Medium" else 0.25)
        
        n_spd = float(current_speeds.get(n, 55.0))
        n_prob = float(current_probs.get(n, 0.0))
        
        if rel.get("is_directional") and not rel.get("is_counter_flow"):
            # n is an upstream feeder to sensor_id
            # Upstream shock exists if n is congested or deteriorating faster than sensor_id
            spd_shock = 1.0 if n_spd < congestion_threshold else max(0.0, curr_spd - n_spd) / 30.0
            upstream_scores.append(conf_weight * (0.6 * n_prob + 0.4 * np.clip(spd_shock, 0.0, 1.0)))
        else:
            # Check counter-direction: sensor_id -> n
            rel_rev = prop_engine.get_relationship(sensor_id, n)
            if rel_rev.get("is_directional") and not rel_rev.get("is_counter_flow"):
                # n is downstream from sensor_id. Downstream bottleneck backpressure:
                down_shock = 1.0 if n_spd < congestion_threshold else max(0.0, n_prob)
                downstream_scores.append(conf_weight * down_shock)
            else:
                # Direction uncertain neighbor
                if n_spd < congestion_threshold:
                    upstream_scores.append(0.25 * n_prob)
                    
    up_val = float(np.max(upstream_scores)) if upstream_scores else 0.0
    down_val = float(np.max(downstream_scores)) if downstream_scores else 0.0
    
    raw_psi = 0.60 * up_val + 0.40 * down_val
    return float(np.clip(raw_psi, 0.0, 1.0))


def compute_all_spatiotemporal_metrics(
    sensor_ids: List[str],
    graph: nx.Graph,
    prop_engine,
    current_speeds: Dict[str, float],
    current_rf_probs: Dict[str, float],
    prev_rf_probs: Optional[Dict[str, float]] = None,
    speed_delta_5: Optional[Dict[str, float]] = None,
    speed_delta_10: Optional[Dict[str, float]] = None,
    rolling_trend_30: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Vectorized/batched computation of TRM, SCP, and PSI for all sensors.
    """
    speed_d5 = speed_delta_5 or {}
    speed_d10 = speed_delta_10 or {}
    trend_30 = rolling_trend_30 or {}
    
    results = {}
    for s in sensor_ids:
        p = current_rf_probs.get(s, 0.0)
        p_prev = prev_rf_probs.get(s, None) if prev_rf_probs else None
        spd = current_speeds.get(s, 55.0)
        d5 = speed_d5.get(s, 0.0)
        d10 = speed_d10.get(s, 0.0)
        tr = trend_30.get(s, 0.0)
        
        trm = compute_temporal_risk_momentum(
            current_prob=p,
            prev_prob=p_prev,
            current_speed=spd,
            speed_delta_5=d5,
            speed_delta_10=d10,
            rolling_trend_30=tr,
        )
        scp = compute_spatial_congestion_pressure(
            sensor_id=s,
            graph=graph,
            current_speeds=current_speeds,
            current_probs=current_rf_probs,
        )
        psi = compute_propagation_shock_index(
            sensor_id=s,
            graph=graph,
            prop_engine=prop_engine,
            current_speeds=current_speeds,
            current_probs=current_rf_probs,
            prev_probs=prev_rf_probs,
        )
        
        results[s] = {
            "TRM": trm,
            "SCP": scp,
            "PSI": psi,
        }
        
    return results
