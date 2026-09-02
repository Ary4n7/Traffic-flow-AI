"""
PR·VIGIL — Predictive Safety Hotspot Aggregator Engine
===========================================================
Spatially aggregates near-miss events, low-TTC incidents, and VRU conflict patterns
to compute a Predictive Safety Hotspot Index across Indian road corridors.
"""

import math
from typing import List, Dict, Tuple

class SafetyHotspotEngine:
    def __init__(self, decay_rate: float = 0.95):
        self.decay_rate = decay_rate
        self.node_risk_scores: Dict[str, float] = {}
        self.node_event_counts: Dict[str, Dict[str, int]] = {}

    def update_node_safety(self, node_id: str, near_misses: List[Dict]) -> float:
        """
        Updates the predictive safety risk score for a specific road node/corridor segment.
        """
        current_score = self.node_risk_scores.get(node_id, 10.0) * self.decay_rate
        
        if node_id not in self.node_event_counts:
            self.node_event_counts[node_id] = {"critical": 0, "high_risk": 0, "vru": 0}
            
        for nm in near_misses:
            risk = nm.get("risk_level", "WATCH")
            vru = nm.get("vru_involved", False)
            
            if risk == "CRITICAL":
                current_score += 25.0
                self.node_event_counts[node_id]["critical"] += 1
            elif risk == "HIGH RISK":
                current_score += 15.0
                self.node_event_counts[node_id]["high_risk"] += 1
                
            if vru:
                current_score += 12.0
                self.node_event_counts[node_id]["vru"] += 1
                
        final_score = round(min(100.0, max(0.0, current_score)), 1)
        self.node_risk_scores[node_id] = final_score
        return final_score

    def get_safety_hotspots(self, node_metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Generates spatial safety hotspot objects formatted for GIS rendering.
        """
        hotspots = []
        for node_id, score in self.node_risk_scores.items():
            meta = node_metadata.get(node_id, {})
            lat = meta.get("lat", 12.9172)
            lon = meta.get("lon", 77.6228)
            name = meta.get("name", node_id)
            
            counts = self.node_event_counts.get(node_id, {"critical": 0, "high_risk": 0, "vru": 0})
            
            if score >= 65.0:
                tier = "🔴 CRITICAL SAFETY HOTSPOT"
            elif score >= 40.0:
                tier = "🟠 HIGH SAFETY RISK HOTSPOT"
            elif score >= 20.0:
                tier = "🟡 WATCH HOTSPOT"
            else:
                tier = "🟢 NORMAL SAFETY ZONE"
                
            hotspots.append({
                "node_id": node_id,
                "node_name": name,
                "lat": lat,
                "lon": lon,
                "safety_score": score,
                "tier": tier,
                "critical_near_misses": counts["critical"],
                "high_risk_near_misses": counts["high_risk"],
                "vru_conflicts": counts["vru"],
                "radius_meters": int(50 + score * 3.5),
            })
            
        return sorted(hotspots, key=lambda x: x["safety_score"], reverse=True)
