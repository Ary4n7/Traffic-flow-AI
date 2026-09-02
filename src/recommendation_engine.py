"""
PR·VIGIL — Proactive AI Response Recommendation Engine
==========================================================
Generates automated, explainable operational mitigation advisories
for traffic control centers, variable message signs (VMS), and emergency services.
Architectural Inspiration: Predict → Visualize → Explain → Recommend.
"""

from typing import List, Dict

class ProactiveRecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(
        self,
        active_near_misses: List[Dict],
        vru_status: Dict,
        congestion_predictions: Dict[str, Dict],
        hotspots: List[Dict]
    ) -> List[Dict]:
        """
        Synthesizes safety intelligence and mobility forecasts to produce prioritized
        proactive countermeasure recommendations.
        """
        recommendations = []
        rec_id = 1
        
        # 1. Immediate VRU Safety Interventions
        if vru_status.get("vru_exposure_level") == "ELEVATED" or any(nm.get("vru_involved") for nm in active_near_misses):
            recommendations.append({
                "id": f"REC-SAF-{rec_id:03d}",
                "priority": "🔴 URGENT / SAFETY",
                "target": "Pedestrian & Two-Wheeler Corridors",
                "title": "Activate Variable Message Sign (VMS) Pedestrian Advisory",
                "reason": "Multiple high-risk VRU interactions and low-TTC near-misses detected on urban corridor.",
                "action": "Flash VMS Caution: 'MIXED TRAFFIC / VRU CROSSING AHEAD — REDUCE SPEED TO 30 KM/H'. Dispatch traffic warden if sustained.",
                "integration": "Connected Traffic Signal Controller (NEMA/TS2) & Smart VMS Network"
            })
            rec_id += 1

        # 2. Critical Near-Miss / Collision Warning Interventions
        critical_nms = [nm for nm in active_near_misses if nm.get("risk_level") == "CRITICAL"]
        if critical_nms:
            nm = critical_nms[0]
            recommendations.append({
                "id": f"REC-SAF-{rec_id:03d}",
                "priority": "🔴 CRITICAL / SAFETY",
                "target": f"Location ({nm.get('location')})",
                "title": "Trigger Operator Incident Watch & Automated CCTV Zoom",
                "reason": f"Active trajectory conflict detected between {nm.get('interaction')} (TTC: {nm.get('ttc_sec')}s).",
                "action": "Auto-focus nearest PTZ traffic camera to intersection. Prompt operator to confirm safety state.",
                "integration": "Traffic Operations Center (TOC) Video Management System"
            })
            rec_id += 1

        # 3. 15-Minute Congestion Spillover Mitigation
        spillover_nodes = []
        for node_id, pred in congestion_predictions.items():
            if pred.get("prob_15min", 0.0) > 0.60:
                spillover_nodes.append(pred.get("node_name", node_id))
                
        if spillover_nodes:
            corridors_str = ", ".join(spillover_nodes[:3])
            recommendations.append({
                "id": f"REC-MOB-{rec_id:03d}",
                "priority": "🟠 HIGH / MOBILITY",
                "target": f"Corridors: {corridors_str}",
                "title": "Deploy Proactive Dynamic Signal Timing Adjustment",
                "reason": "ML Model forecasts >60% probability of secondary spillover congestion within 10–15 minutes.",
                "action": f"Extend green-split timing by +12s on outbound arterial routes to flush bottleneck at {spillover_nodes[0]}.",
                "integration": "Adaptive Traffic Signal Control System (ATCS / SCATS / SCOOT)"
            })
            rec_id += 1

        # 4. Spatial Safety Hotspot Review
        high_hotspots = [h for h in hotspots if h.get("safety_score", 0) >= 40.0]
        if high_hotspots:
            hs = high_hotspots[0]
            recommendations.append({
                "id": f"REC-ENG-{rec_id:03d}",
                "priority": "🟡 MODERATE / ENGINEERING",
                "target": f"Hotspot: {hs.get('node_name')}",
                "title": "Schedule Road Infrastructure & Speed Enforcement Review",
                "reason": f"Deteriorating predictive safety score ({hs.get('safety_score')}/100) due to {hs.get('critical_near_misses')} near-misses.",
                "action": "Issue automated engineering work order for road marking audit, speed-bump inspection, and police patrol positioning.",
                "integration": "Municipal Road Authority Work Order & Traffic Police Enforcement System"
            })
            rec_id += 1

        # Fallback recommendation if clean state
        if not recommendations:
            recommendations.append({
                "id": "REC-SYS-001",
                "priority": "🟢 NORMAL / MONITORING",
                "target": "Entire Road Network",
                "title": "Standard AI Patrol & Continuous Telemetry Sweep",
                "reason": "Road safety indices and 15-minute mobility forecasts are within nominal operating parameters.",
                "action": "Maintain automated camera monitoring and real-time TTC interaction sweeps.",
                "integration": "PR·VIGIL Autonomous Monitoring Core"
            })

        return recommendations
