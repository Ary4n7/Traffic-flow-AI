"""
PR·VIGIL — Vulnerable Road User (VRU) Safety Engine
======================================================
Dedicated safety analysis engine for Pedestrians, Cyclists, and Two-Wheelers
in mixed Indian road traffic environments.
"""

import math
from typing import Dict, List, Tuple
from src.cv_safety_pipeline import TrackedObject

VRU_WEIGHTS = {
    "pedestrian": 2.5,
    "bicycle": 2.0,
    "motorcycle": 1.8,
    "auto_rickshaw": 1.3,
    "car": 1.0,
    "bus": 1.0,
    "truck": 1.0,
}

class VRUSafetyEngine:
    def __init__(self, critical_ttc_threshold: float = 1.5):
        self.critical_ttc_threshold = critical_ttc_threshold

    def calculate_vru_risk_score(self, obj1: TrackedObject, obj2: TrackedObject, ttc: float, distance_m: float) -> Tuple[float, str, str]:
        """
        Computes a normalized VRU Safety Risk Score [0.0 to 100.0] and risk category.
        Returns: (risk_score, risk_category, detailed_explanation)
        """
        w1 = VRU_WEIGHTS.get(obj1.class_name.lower(), 1.0)
        w2 = VRU_WEIGHTS.get(obj2.class_name.lower(), 1.0)
        vru_multiplier = max(w1, w2)
        
        # Exponential TTC decay factor
        ttc_factor = math.exp(-0.8 * max(0.1, ttc))
        
        # Proximity decay factor
        proximity_factor = max(0.0, 1.0 - (distance_m / 15.0))
        
        raw_score = (0.6 * ttc_factor + 0.4 * proximity_factor) * 100.0 * (vru_multiplier / 2.5)
        risk_score = round(min(100.0, max(0.0, raw_score)), 1)
        
        # Categorize VRU Risk Level
        if risk_score >= 80.0:
            category = "CRITICAL VRU RISK"
        elif risk_score >= 60.0:
            category = "HIGH VRU RISK"
        elif risk_score >= 35.0:
            category = "MODERATE VRU RISK"
        else:
            category = "LOW VRU RISK"
            
        vru_type = obj1.class_name if obj1.is_vru else obj2.class_name
        veh_type = obj2.class_name if obj1.is_vru else obj1.class_name
        
        explanation = (
            f"{category}: Interacting {vru_type.upper()} and {veh_type.upper()} "
            f"with TTC of {ttc:.1f}s at {distance_m:.1f}m distance. VRU Multiplier: {vru_multiplier}x."
        )
        
        return risk_score, category, explanation

    def summarize_vru_status(self, active_objects: List[TrackedObject]) -> Dict:
        """
        Provides summary statistics of active VRUs on the road.
        """
        vru_objects = [o for o in active_objects if o.is_vru]
        pedestrians = [o for o in vru_objects if o.class_name == "pedestrian"]
        two_wheelers = [o for o in vru_objects if o.class_name == "motorcycle"]
        
        return {
            "total_vrus_detected": len(vru_objects),
            "pedestrians_count": len(pedestrians),
            "two_wheelers_count": len(two_wheelers),
            "vru_exposure_level": "ELEVATED" if len(vru_objects) >= 3 else "NORMAL"
        }
