"""
src/simulator.py
----------------
Traffic Simulator Engine for TrafficFlow AI.
Supports full 24-hour simulation timeline (288 steps of 5-min intervals: 00:00 to 23:55).

Simulates dynamic traffic conditions across 4 primary options:
1. Normal Flow: Baseline highway flow with natural stochastic variations.
2. Morning Rush Hour: Peak morning commute surge (07:00–10:00).
3. Evening Rush Hour: Peak evening commute surge (16:00–20:00).
4. Bottleneck Incident: User-triggered accident/choke scenario on target highway segments.
"""

import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.predictor import SpilloverPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("simulator")


class TrafficSimulator:
    TOTAL_STEPS_PER_DAY = 288  # 24 hours * 12 (5-min steps) = 288 steps (0 to 287)

    def __init__(
        self,
        graph_path: str = config.ADJACENCY_PKL,
        predictor: SpilloverPredictor = None,
    ):
        """Initialize simulation environment with 207 road network nodes."""
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph not found at {graph_path}.")
            
        import pickle
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)
            
        self.sensor_ids = list(self.graph.nodes())
        self.predictor = predictor
        
        # State variables
        self.start_timestamp = pd.Timestamp("2026-10-15 00:00:00")
        self.current_time = self.start_timestamp
        self.active_incidents = {} # road_id -> {severity: 'High', step_count: 0}
        self.history_buffer = [] # list of pd.Series (past speeds)
        self.step_count = 0
        self.step_history = [] # list of full step result dicts
        self.current_regime = "normal"
        
        # Initialize normal baseline speeds (58 - 66 mph)
        np.random.seed(42)
        self.current_speeds = pd.Series(
            data=np.random.uniform(58.0, 66.0, size=len(self.sensor_ids)).astype(np.float32),
            index=self.sensor_ids,
        )
        
        # Warmup history buffer with 6 initial steps
        for _ in range(6):
            self.history_buffer.append(self.current_speeds.copy())
            
        # Record initial Step 0 snapshot (00:00)
        self._record_step_0()

    def _record_step_0(self):
        """Record baseline Step 0 snapshot at 00:00 AM."""
        history_df = pd.DataFrame(self.history_buffer)
        ai_predictions = None
        if self.predictor is not None:
            ai_predictions = self.predictor.predict_network(
                current_speeds=self.current_speeds,
                speed_history=history_df,
                timestamp=self.current_time,
            )
        congested_count = int((self.current_speeds < config.CONGESTION_THRESHOLD_MPH).sum())
        initial_snap = {
            "timestamp": self.current_time,
            "speeds": self.current_speeds.copy(),
            "history": history_df.copy(),
            "congested_count": congested_count,
            "active_incidents": list(self.active_incidents.keys()),
            "ai_predictions": ai_predictions,
            "step_count": 0,
            "regime": self.current_regime,
        }
        self.step_history = [initial_snap]

    def trigger_incident(self, road_id: str, severity: str = "High"):
        """Inject an accident/incident on a specific road."""
        road_id = str(road_id)
        if road_id in self.sensor_ids:
            self.active_incidents[road_id] = {
                "severity": severity,
                "step_count": 0,
            }
            drop_speed = 10.0 if severity == "High" else (16.0 if severity == "Medium" else 22.0)
            self.current_speeds[road_id] = drop_speed
            log.info("🚨 Incident triggered on Road %s (%s severity: speed %.1f mph)", road_id, severity, drop_speed)

    def clear_incidents(self):
        """Reset all active incidents."""
        self.active_incidents.clear()

    def reset_network(self):
        """Restore the simulator to its original initial Step 0 state at 00:00 AM."""
        self.current_time = self.start_timestamp
        self.active_incidents.clear()
        self.history_buffer.clear()
        self.step_count = 0
        self.current_regime = "normal"
        
        np.random.seed(42)
        self.current_speeds = pd.Series(
            data=np.random.uniform(58.0, 66.0, size=len(self.sensor_ids)).astype(np.float32),
            index=self.sensor_ids,
        )
        
        for _ in range(6):
            self.history_buffer.append(self.current_speeds.copy())
            
        self._record_step_0()
        log.info("↺ Simulator reset to initial 00:00 AM network state.")
        return self.step_history[0]

    def reset(self):
        """Reset network state (alias for reset_network)."""
        return self.reset_network()

    def get_step(self, target_step: int, regime: str = None) -> dict:
        """Get snapshot for target_step (0 to 287). Simulates forward if necessary."""
        target_step = max(0, min(target_step, self.TOTAL_STEPS_PER_DAY - 1))
        
        if target_step < len(self.step_history):
            snap = self.step_history[target_step]
            if snap.get("ai_predictions") is None and self.predictor is not None:
                snap["ai_predictions"] = self.predictor.predict_network(
                    current_speeds=snap["speeds"],
                    speed_history=snap["history"],
                    timestamp=snap["timestamp"],
                )
            return snap
        
        # Step forward until target_step is reached
        while len(self.step_history) <= target_step:
            compute_pred = (len(self.step_history) == target_step)
            self.step(regime=regime or self.current_regime, compute_predictions=compute_pred)
            
        return self.step_history[target_step]

    def induce_incident(self, road_id: str, target_speed: float = 12.0, duration_steps: int = 4) -> dict:
        """Induce synthetic incident speed drop on a target sensor and return step state."""
        road_id = str(road_id)
        if road_id in self.sensor_ids:
            self.active_incidents[road_id] = {
                "severity": "High",
                "step_count": 0,
            }
            self.current_speeds[road_id] = target_speed
            log.info("🚨 Incident induced on Road %s (target speed: %.1f mph)", road_id, target_speed)
        return self.step(regime="incident", compute_predictions=True)

    def step(self, traffic_intensity: float = 0.5, propagation_rate: float = 0.4, regime: str = None, compute_predictions: bool = True) -> dict:
        """
        Advance the simulation by one 5-minute interval across the 24-hour timeline.
        """
        if regime:
            self.current_regime = regime
            
        self.step_count += 1
        self.current_time = self.start_timestamp + pd.Timedelta(minutes=self.step_count * 5)
        new_speeds = self.current_speeds.copy()
        
        hour = self.current_time.hour
        
        is_morning_peak = (7 <= hour <= 9) or (self.current_regime == "morning_rush")
        is_evening_peak = (16 <= hour <= 19) or (self.current_regime == "evening_rush")
        is_incident_mode = (self.current_regime == "incident") or len(self.active_incidents) > 0
        
        base_noise = np.random.normal(0.0, 1.0, size=len(self.sensor_ids))
        
        corridor_sensors = self.sensor_ids[:45]
        severe_corridor = self.sensor_ids[10:28]
        
        for s in self.sensor_ids:
            if s not in self.active_incidents:
                if is_incident_mode and s in severe_corridor:
                    # Severe bottleneck incident: rapid deceleration down to 12-16 mph (HIGH RISK & CRITICAL)
                    idx = severe_corridor.index(s)
                    drop_amount = min(48.0, 12.0 + (idx * 2.0) + (self.step_count % 8) * 4.0)
                    target_speed = max(11.0, 62.0 - drop_amount)
                elif is_morning_peak and s in corridor_sensors:
                    # Sustained morning commute deceleration wave (WATCH -> WARNING -> HIGH RISK -> CRITICAL)
                    idx = corridor_sensors.index(s)
                    drop_amount = min(44.0, 10.0 + (idx * 0.8) + (self.step_count % 12) * 2.8)
                    target_speed = max(14.0, 62.0 - drop_amount)
                elif is_evening_peak and s in corridor_sensors:
                    # Sustained evening commute deceleration wave
                    idx = corridor_sensors.index(s)
                    drop_amount = min(42.0, 8.0 + (idx * 0.7) + (self.step_count % 12) * 2.6)
                    target_speed = max(15.0, 62.0 - drop_amount)
                else:
                    target_speed = 63.0
                    
                updated_val = float(new_speeds[s] * 0.55 + target_speed * 0.45 + base_noise[self.sensor_ids.index(s)])
                new_speeds[s] = np.float32(updated_val)
                
        # 2. Process active incidents & propagate spillover
        for inc_road, data in list(self.active_incidents.items()):
            data["step_count"] += 1
            severity = data["severity"]
            base_drop = 8.0 if severity == "High" else 15.0
            new_speeds[inc_road] = base_drop + np.random.uniform(-1.5, 1.5)
            
            # Propagate congestion to 1-hop neighbors
            for neighbor in self.graph.neighbors(inc_road):
                if neighbor not in self.active_incidents:
                    decay_factor = min(0.65, 0.2 * data["step_count"] * propagation_rate)
                    target_neigh_speed = max(14.0, new_speeds[neighbor] * (1.0 - decay_factor))
                    new_speeds[neighbor] = new_speeds[neighbor] * 0.5 + target_neigh_speed * 0.5
                    
            # Propagate to 2-hop neighbors after 2 steps
            if data["step_count"] >= 2:
                for n1 in self.graph.neighbors(inc_road):
                    for n2 in self.graph.neighbors(n1):
                        if n2 != inc_road and n2 not in self.graph.neighbors(inc_road):
                            new_speeds[n2] = max(22.0, new_speeds[n2] * 0.82)
                            
        # Clip speeds to realistic highway bounds
        new_speeds = new_speeds.clip(lower=5.0, upper=75.0)
        self.current_speeds = new_speeds
        
        # Update history
        self.history_buffer.append(self.current_speeds.copy())
        if len(self.history_buffer) > 12:
            self.history_buffer.pop(0)
            
        history_df = pd.DataFrame(self.history_buffer)
        
        # 3. Compute live AI Predictions if requested and predictor attached
        ai_predictions = None
        if compute_predictions and self.predictor is not None:
            ai_predictions = self.predictor.predict_network(
                current_speeds=self.current_speeds,
                speed_history=history_df,
                timestamp=self.current_time,
            )
            
        congested_count = int((self.current_speeds < config.CONGESTION_THRESHOLD_MPH).sum())
        
        res = {
            "timestamp": self.current_time,
            "speeds": self.current_speeds.copy(),
            "history": history_df.copy(),
            "congested_count": congested_count,
            "active_incidents": list(self.active_incidents.keys()),
            "ai_predictions": ai_predictions,
            "step_count": self.step_count,
            "regime": self.current_regime,
        }
        self.step_history.append(res)
        return res
