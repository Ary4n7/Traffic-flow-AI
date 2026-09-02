"""
PR·VIGIL — Indian Geospatial (GIS) Map & Overlay Generator (v2 State-Bound Architecture)
========================================================================================
Supports dynamic green/yellow/orange/red speed corridor lines, city-specific accident hotspots,
and dedicated alternate route map visualizers. Dynamic safety risk scores per node.

Speed Color Semantics (Sanity Verified):
- 🟢 Green (#00E676): speed > 35 km/h (Free-flow / Good flow)
- 🟡 Yellow (#FFEA00): 25-35 km/h (Moderate speed)
- 🟠 Orange (#FF9100): 15-25 km/h (Slow speed)
- 🔴 Red (#FF1744): < 15 km/h (Congested / Heavy bottleneck)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple

ScatterMapClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
if ScatterMapClass is None:
    ScatterMapClass = go.Scattergeo

# Representative Indian Road Corridor Metadata
INDIAN_ROAD_NODES = {
    # Bengaluru Hub (Silk Board & Outer Ring Road)
    "IND_BLR_01": {"name": "Silk Board Junction", "lat": 12.9172, "lon": 77.6228, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_02": {"name": "HSR Layout 14th Main", "lat": 12.9116, "lon": 77.6389, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_03": {"name": "Agara Lake Junction", "lat": 12.9258, "lon": 77.6483, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_04": {"name": "Iblur Flyover", "lat": 12.9289, "lon": 77.6681, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_05": {"name": "Bellandur EcoSpace", "lat": 12.9272, "lon": 77.6835, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_06": {"name": "Kadubeesanahalli", "lat": 12.9356, "lon": 77.6974, "city": "Bengaluru", "corridor": "Bengaluru ORR"},
    "IND_BLR_07": {"name": "Marathahalli Bridge", "lat": 12.9562, "lon": 77.7019, "city": "Bengaluru", "corridor": "Bengaluru ORR"},

    # Delhi NCR Hub (DND Flyway & Ring Road)
    "IND_DEL_01": {"name": "Ashram Chowk", "lat": 28.5708, "lon": 77.2583, "city": "Delhi", "corridor": "Delhi Ring Road"},
    "IND_DEL_02": {"name": "Lajpat Nagar Metro Junction", "lat": 28.5694, "lon": 77.2417, "city": "Delhi", "corridor": "Delhi Ring Road"},
    "IND_DEL_03": {"name": "AIIMS Flyover Junction", "lat": 28.5672, "lon": 77.2100, "city": "Delhi", "corridor": "Delhi Ring Road"},
    "IND_DEL_04": {"name": "DND Flyway Toll Plaza", "lat": 28.5678, "lon": 77.2831, "city": "Delhi", "corridor": "DND Expressway"},
    "IND_DEL_05": {"name": "Mayur Vihar Phase 1", "lat": 28.6041, "lon": 77.2942, "city": "Delhi", "corridor": "Noida Link"},

    # Mumbai Hub (Western Express Highway & BKC)
    "IND_BOM_01": {"name": "Bandra-Kurla Complex (BKC) Entry", "lat": 19.0657, "lon": 72.8686, "city": "Mumbai", "corridor": "BKC Connector"},
    "IND_BOM_02": {"name": "Kalanagar Junction Bandra", "lat": 19.0601, "lon": 72.8519, "city": "Mumbai", "corridor": "Mumbai WEH"},
    "IND_BOM_03": {"name": "Dadar TT Circle", "lat": 19.0178, "lon": 72.8478, "city": "Mumbai", "corridor": "Dr. BA Road"},
    "IND_BOM_04": {"name": "WEH Kurla Slip Road", "lat": 19.0700, "lon": 72.8600, "city": "Mumbai", "corridor": "Mumbai WEH"},
}

# City-Specific Historical Accident Hotspots
CITY_ACCIDENT_HOTSPOTS = {
    "Bengaluru": [
        {"name": "Silk Board Junction", "lat": 12.9172, "lon": 77.6228, "reason": "High pedestrian crossing volume near market & bus stop", "incidents": 38},
        {"name": "Iblur Flyover Merge", "lat": 12.9289, "lon": 77.6681, "reason": "Merge point with no dedicated lane for auto-rickshaws", "incidents": 29},
        {"name": "Marathahalli Bridge Curve", "lat": 12.9562, "lon": 77.7019, "reason": "Frequent wrong-side overtaking by two-wheelers", "incidents": 34},
        {"name": "Agara Lake Junction", "lat": 12.9258, "lon": 77.6483, "reason": "Poor visibility at unsignaled high-speed median crossing", "incidents": 22},
    ],
    "Delhi": [
        {"name": "Ashram Chowk Bottleneck", "lat": 28.5708, "lon": 77.2583, "reason": "High-density multi-corridor weaving & pedestrian crossing", "incidents": 42},
        {"name": "Lajpat Nagar Ring Road", "lat": 28.5694, "lon": 77.2417, "reason": "Jaywalking volume near metro station with high vehicle speeds", "incidents": 31},
        {"name": "DND Flyway Toll Approach", "lat": 28.5678, "lon": 77.2831, "reason": "High-speed lane splitting by two-wheelers near toll booths", "incidents": 27},
        {"name": "AIIMS Flyover Underpass", "lat": 28.5672, "lon": 77.2100, "reason": "Narrow underpass merge with emergency ambulance traffic", "incidents": 36},
    ],
    "Mumbai": [
        {"name": "BKC Connector Entry", "lat": 19.0657, "lon": 72.8686, "reason": "Sharp curve weaving between heavy buses & two-wheelers", "incidents": 35},
        {"name": "Kalanagar Junction Bandra", "lat": 19.0601, "lon": 72.8519, "reason": "Heavy weaving traffic from WEH elevated flyover slip road", "incidents": 40},
        {"name": "Dadar TT Circle", "lat": 19.0178, "lon": 72.8478, "reason": "Unsignaled pedestrian market crossing with high commercial density", "incidents": 33},
        {"name": "WEH Kurla Slip Road", "lat": 19.0700, "lon": 72.8600, "reason": "Speed differential between cars & motorized auto-rickshaws", "incidents": 28},
    ]
}


def get_speed_color(speed_kmh: float) -> Tuple[str, str]:
    if speed_kmh > 35.0:
        return "#00E676", "🟢 Free (>35 km/h)"
    elif speed_kmh >= 25.0:
        return "#FFEA00", "🟡 Slow (25-35 km/h)"
    elif speed_kmh >= 15.0:
        return "#FF9100", "🟠 Mid Congested (15-25 km/h)"
    else:
        return "#FF1744", "🔴 Heavy Congested (<15 km/h)"


def verify_speed_color_legend():
    c_green, l_green = get_speed_color(42.0)
    c_yellow, l_yellow = get_speed_color(30.0)
    c_orange, l_orange = get_speed_color(20.0)
    c_red, l_red = get_speed_color(10.0)
    
    assert c_green == "#00E676", f"Expected Green #00E676, got {c_green}"
    assert c_yellow == "#FFEA00", f"Expected Yellow #FFEA00, got {c_yellow}"
    assert c_orange == "#FF9100", f"Expected Orange #FF9100, got {c_orange}"
    assert c_red == "#FF1744", f"Expected Red #FF1744, got {c_red}"
    return True

verify_speed_color_legend()


class IndiaGISMapper:
    def __init__(self, city_hub: str = "Bengaluru"):
        self.city_hub = city_hub

    def build_gis_figure(
        self,
        node_speeds: Dict[str, float],
        safety_hotspots: List[Dict],
        near_misses: List[Dict],
        congestion_predictions: Dict[str, Dict],
        mapbox_token: str = None
    ) -> go.Figure:
        fig = go.Figure()
        
        if "Delhi" in self.city_hub:
            target_city = "Delhi"
            center_lat, center_lon, zoom = 28.5720, 77.2550, 12.2
        elif "Mumbai" in self.city_hub:
            target_city = "Mumbai"
            center_lat, center_lon, zoom = 19.0500, 72.8550, 12.2
        else:
            target_city = "Bengaluru"
            center_lat, center_lon, zoom = 12.9220, 77.6550, 12.1

        city_nodes = {k: v for k, v in INDIAN_ROAD_NODES.items() if v["city"] == target_city}
        if not city_nodes:
            city_nodes = INDIAN_ROAD_NODES

        # Speed Color Category Traces
        color_groups = {
            "green": {"color": "#00E676", "name": "🟢 Free (>35 km/h)", "lats": [], "lons": [], "texts": []},
            "yellow": {"color": "#FFEA00", "name": "🟡 Slow (25-35 km/h)", "lats": [], "lons": [], "texts": []},
            "orange": {"color": "#FF9100", "name": "🟠 Mid Congested (15-25 km/h)", "lats": [], "lons": [], "texts": []},
            "red": {"color": "#FF1744", "name": "🔴 Heavy Congested (<15 km/h)", "lats": [], "lons": [], "texts": []},
        }

        node_keys = list(city_nodes.keys())
        for i in range(len(node_keys) - 1):
            n1 = city_nodes[node_keys[i]]
            n2 = city_nodes[node_keys[i+1]]
            speed1 = node_speeds.get(node_keys[i], 42.0)
            speed2 = node_speeds.get(node_keys[i+1], 42.0)
            avg_speed = (speed1 + speed2) / 2.0
            
            hex_color, label = get_speed_color(avg_speed)
            if hex_color == "#00E676":
                grp = color_groups["green"]
            elif hex_color == "#FFEA00":
                grp = color_groups["yellow"]
            elif hex_color == "#FF9100":
                grp = color_groups["orange"]
            else:
                grp = color_groups["red"]

            grp["lats"].extend([n1["lat"], n2["lat"], None])
            grp["lons"].extend([n1["lon"], n2["lon"], None])
            grp["texts"].extend([
                f"Corridor Link: {n1['name']} ↔ {n2['name']}<br>Speed: {avg_speed:.1f} km/h ({label})",
                f"Corridor Link: {n1['name']} ↔ {n2['name']}<br>Speed: {avg_speed:.1f} km/h ({label})",
                ""
            ])

        for cat, data in color_groups.items():
            if data["lats"]:
                fig.add_trace(ScatterMapClass(
                    lat=data["lats"],
                    lon=data["lons"],
                    mode="lines",
                    line=dict(width=7, color=data["color"]),
                    hoverinfo="text",
                    hovertext=data["texts"],
                    name=data["name"]
                ))

        # Telemetry Sensor Markers
        lats, lons, texts, colors, sizes = [], [], [], [], []
        for node_id, info in city_nodes.items():
            lats.append(info["lat"])
            lons.append(info["lon"])
            spd = node_speeds.get(node_id, 42.0)
            
            pred_15 = congestion_predictions.get(node_id, {}).get("prob_15min", 0.15) * 100.0
            hex_color, label = get_speed_color(spd)
            colors.append(hex_color)
            sizes.append(18)
            
            hover_text = (
                f"<b>{info['name']}</b> ({node_id})<br>"
                f"Corridor: {info['corridor']}<br>"
                f"Current Speed: {spd:.1f} km/h ({label})<br>"
                f"Predicted 15-Min Congestion: {pred_15:.1f}%"
            )
            texts.append(hover_text)

        fig.add_trace(ScatterMapClass(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, opacity=0.95),
            text=[n["name"].split()[0] for n in city_nodes.values()],
            textposition="top center",
            textfont=dict(size=11, color="#FFFFFF"),
            hoverinfo="text",
            hovertext=texts,
            name="Telemetry Sensors"
        ))

        # City-Specific Historical Accident Hotspot Layer
        city_hotspots = CITY_ACCIDENT_HOTSPOTS.get(target_city, CITY_ACCIDENT_HOTSPOTS["Bengaluru"])
        ah_lats, ah_lons, ah_texts = [], [], []
        for ah in city_hotspots:
            ah_lats.append(ah["lat"])
            ah_lons.append(ah["lon"])
            ah_texts.append(
                f"<b>⚠️ HISTORICAL ACCIDENT HOTSPOT ({target_city})</b><br>"
                f"Location: <b>{ah['name']}</b><br>"
                f"Hazard Reason: <i>{ah['reason']}</i><br>"
                f"Annual Reported Incidents: {ah['incidents']}"
            )

        fig.add_trace(ScatterMapClass(
            lat=ah_lats,
            lon=ah_lons,
            mode="markers",
            marker=dict(size=24, color="#D97706", symbol="circle", opacity=0.9),
            hoverinfo="text",
            hovertext=ah_texts,
            name=f"Historical Accident Hotspots ({target_city})"
        ))

        # Scaling Live Near-Miss Hotspots with Dynamic Safety Scores (Range 18 to 92)
        if safety_hotspots:
            hs_lats, hs_lons, hs_sizes, hs_texts = [], [], [], []
            for hs in safety_hotspots:
                hs_lats.append(hs.get("lat", list(city_nodes.values())[0]["lat"]))
                hs_lons.append(hs.get("lon", list(city_nodes.values())[0]["lon"]))
                node_id_hs = hs.get("node_id", "")
                node_spd = node_speeds.get(node_id_hs, 30.0) if node_speeds else 30.0
                
                # Dynamic Safety Risk Score based on active node speed & near-miss count
                crit_nm = hs.get("critical_near_misses", 1)
                dyn_score = max(18.0, min(94.0, 96.0 - (node_spd / 45.0) * 52.0 + (crit_nm * 6.0)))
                
                sz = int(28 + dyn_score * 0.45)
                hs_sizes.append(sz)
                hs_texts.append(
                    f"<b>🔴 LIVE RISK: ACCIDENT & NEAR-MISS HOTSPOT</b><br>"
                    f"Location: {hs.get('node_name', 'Corridor Junction')}<br>"
                    f"Current Speed: {node_spd:.1f} km/h<br>"
                    f"Dynamic Safety Risk Index: <b>{dyn_score:.0f}/100</b><br>"
                    f"Critical Near-Misses: {crit_nm}<br>"
                    f"VRU Conflicts: {hs.get('vru_conflicts', 3)}"
                )
                
            fig.add_trace(ScatterMapClass(
                lat=hs_lats,
                lon=hs_lons,
                mode="markers",
                marker=dict(size=hs_sizes, color="#FF1744", opacity=0.45),
                hoverinfo="text",
                hovertext=hs_texts,
                name="Live Active Risk Hotspots"
            ))

        map_config = dict(
            style="carto-darkmatter" if hasattr(go, "Scattermap") else "open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        )

        layout_kwargs = dict(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0B0F17",
            plot_bgcolor="#0B0F17",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top", y=0.98,
                xanchor="left", x=0.01,
                font=dict(color="#FFFFFF", size=11),
                bgcolor="rgba(15, 23, 42, 0.9)",
                bordercolor="#334155",
                borderwidth=1
            ),
            height=540
        )
        
        if hasattr(go, "Scattermap") and ScatterMapClass == getattr(go, "Scattermap", None):
            layout_kwargs["map"] = map_config
        else:
            layout_kwargs["mapbox"] = map_config

        fig.update_layout(**layout_kwargs)
        return fig


def build_alternate_route_figure(city_hub: str, is_congested: bool = True) -> go.Figure:
    fig = go.Figure()
    
    if "Delhi" in city_hub:
        target_city = "Delhi"
        center_lat, center_lon, zoom = 28.5720, 77.2350, 12.5
        primary_name = "Ashram Chowk Ring Road Corridor"
        bypass_name = "Barapullah Elevated Bypass Corridor"
        p_lats = [28.5708, 28.5694, 28.5672]
        p_lons = [77.2583, 77.2417, 77.2100]
        b_lats = [28.5708, 28.5830, 28.5745, 28.5672]
        b_lons = [77.2583, 77.2480, 77.2280, 77.2100]
    elif "Mumbai" in city_hub:
        target_city = "Mumbai"
        center_lat, center_lon, zoom = 19.0500, 72.8550, 12.1
        primary_name = "BKC Connector / Kalanagar Junction"
        bypass_name = "Western Express Overhead Flyover Bypass"
        p_lats = [19.0657, 19.0601, 19.0178]
        p_lons = [72.8686, 72.8519, 72.8478]
        b_lats = [19.0657, 19.0700, 19.0178]
        b_lons = [72.8686, 72.8600, 72.8478]
    else:  # Bengaluru
        target_city = "Bengaluru"
        center_lat, center_lon, zoom = 12.9220, 77.6550, 12.0
        primary_name = "Silk Board Outer Ring Road Corridor"
        bypass_name = "Agara-Sarjapur Bypass Arterial"
        p_lats = [12.9172, 12.9116, 12.9258, 12.9289]
        p_lons = [77.6228, 77.6389, 77.6483, 77.6681]
        b_lats = [12.9172, 12.9348, 12.9289]
        b_lons = [77.6228, 77.6245, 77.6681]

    # Primary Congested Route (Red)
    fig.add_trace(ScatterMapClass(
        lat=p_lats,
        lon=p_lons,
        mode="lines+markers",
        line=dict(width=7, color="#FF1744"),
        marker=dict(size=12, color="#FF1744"),
        name=f"🔴 Primary Route: {primary_name} (Heavy Congestion & VRU Risk)",
        hovertext=f"Primary Route: {primary_name} (Bottlenecked)"
    ))

    # Recommended Bypass Route (Green)
    fig.add_trace(ScatterMapClass(
        lat=b_lats,
        lon=b_lons,
        mode="lines+markers",
        line=dict(width=7, color="#00E676"),
        marker=dict(size=12, color="#00E676"),
        name=f"🟢 Recommended Bypass: {bypass_name} (Free-Flow)",
        hovertext=f"Recommended Alternate: {bypass_name}"
    ))

    map_config = dict(
        style="carto-darkmatter" if hasattr(go, "Scattermap") else "open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom
    )

    layout_kwargs = dict(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0B0F17",
        plot_bgcolor="#0B0F17",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            font=dict(color="#FFFFFF", size=11),
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="#334155",
            borderwidth=1
        ),
        height=380
    )
    if hasattr(go, "Scattermap") and ScatterMapClass == getattr(go, "Scattermap", None):
        layout_kwargs["map"] = map_config
    else:
        layout_kwargs["mapbox"] = map_config

    fig.update_layout(**layout_kwargs)
    return fig
