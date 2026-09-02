"""
PR·VIGIL — Computer Vision Road Safety Pipeline & Synthetic Frame Generator
========================================================================================
Supports YOLOv8 detection, multi-object centroid tracking, trajectory extrapolation,
Time-To-Collision (TTC) surrogate safety assessment, and distinct frame generation for Scenarios 1-6.
"""

import math
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class TrackedObject:
    def __init__(self, object_id: int, class_name: str, bbox: Tuple[int, int, int, int]):
        self.object_id = object_id
        self.class_name = class_name
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.cx = (bbox[0] + bbox[2]) / 2.0
        self.cy = (bbox[1] + bbox[3]) / 2.0
        self.history: List[Tuple[float, float]] = [(self.cx, self.cy)]
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed_kmh = 0.0
        self.last_updated = time.time()

    @property
    def is_vru(self) -> bool:
        return self.class_name in ["pedestrian", "motorcycle", "auto_rickshaw"]

    def update_position(self, new_bbox: Tuple[int, int, int, int], dt: float, pixel_to_meter: float = 0.05):
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0
        
        if dt > 0:
            vx_px = (new_cx - self.cx) / dt
            vy_px = (new_cy - self.cy) / dt
            self.velocity_x = 0.7 * self.velocity_x + 0.3 * vx_px
            self.velocity_y = 0.7 * self.velocity_y + 0.3 * vy_px
            speed_ms = math.sqrt(self.velocity_x**2 + self.velocity_y**2) * pixel_to_meter
            self.speed_kmh = max(5.0, speed_ms * 3.6)

        self.bbox = new_bbox
        self.cx = new_cx
        self.cy = new_cy
        self.history.append((self.cx, self.cy))
        if len(self.history) > 30:
            self.history.pop(0)
        self.last_updated = time.time()


class NearMissEvent:
    def __init__(
        self,
        event_id: str,
        obj1: TrackedObject,
        obj2: TrackedObject,
        ttc: float,
        risk_level: str,
        reason: str,
        location: str = "Silk Board Junction"
    ):
        self.event_id = event_id
        self.timestamp = time.strftime("%H:%M:%S")
        self.obj1_id = obj1.object_id
        self.obj1_class = obj1.class_name
        self.obj2_id = obj2.object_id
        self.obj2_class = obj2.class_name
        self.ttc = round(ttc, 2)
        self.risk_level = risk_level
        self.reason = reason
        self.location = location
        self.vru_involved = obj1.class_name in ["pedestrian", "motorcycle"] or obj2.class_name in ["pedestrian", "motorcycle"]

    @property
    def interaction(self) -> str:
        return f"{self.obj1_class.title()} #{self.obj1_id} ↔ {self.obj2_class.title()} #{self.obj2_id}"

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "interaction": self.interaction,
            "ttc_sec": self.ttc,
            "risk_level": self.risk_level,
            "vru_involved": self.vru_involved,
            "reason": self.reason,
            "location": self.location,
        }


class CVSafetyPipeline:
    def __init__(self, fps: float = 30.0, pixel_to_meter: float = 0.05):
        self.fps = fps
        self.dt = 1.0 / fps
        self.pixel_to_meter = pixel_to_meter
        self.next_object_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.near_miss_history: List[NearMissEvent] = []
        self.event_counter = 1

    def calculate_ttc(self, obj1: TrackedObject, obj2: TrackedObject) -> Tuple[float, float, bool]:
        dx = (obj2.cx - obj1.cx) * self.pixel_to_meter
        dy = (obj2.cy - obj1.cy) * self.pixel_to_meter
        distance = math.sqrt(dx**2 + dy**2)
        
        dvx = (obj1.velocity_x - obj2.velocity_x) * self.pixel_to_meter
        dvy = (obj1.velocity_y - obj2.velocity_y) * self.pixel_to_meter
        
        closing_speed = (dx * dvx + dy * dvy) / distance if distance > 0.001 else 0.0
        converging = closing_speed > 0.05
        
        ttc = (distance / closing_speed) if (converging and closing_speed > 0.01) else 99.0
        return max(0.1, ttc), distance, converging

    def assess_risk(self, ttc: float, vru_involved: bool) -> str:
        if ttc < 1.2 or (vru_involved and ttc < 1.4):
            return "CRITICAL"
        elif ttc < 1.8 or (vru_involved and ttc < 2.0):
            return "HIGH RISK"
        elif ttc < 2.8:
            return "WARNING"
        else:
            return "SAFE"

    def process_frame_objects(self, detected_boxes: List[Dict]) -> Tuple[List[TrackedObject], List[NearMissEvent]]:
        current_time = time.time()
        active_objects = []
        
        for det in detected_boxes:
            bbox = det["bbox"]
            class_name = ["car", "motorcycle", "bus", "truck", "auto_rickshaw", "pedestrian"][det["class_id"]]
            
            matched_id = None
            for obj_id, obj in self.tracked_objects.items():
                if obj.class_name == class_name:
                    dist = math.sqrt((obj.cx - (bbox[0]+bbox[2])/2.0)**2 + (obj.cy - (bbox[1]+bbox[3])/2.0)**2)
                    if dist < 65.0:
                        matched_id = obj_id
                        break
                        
            if matched_id is not None:
                obj = self.tracked_objects[matched_id]
                obj.update_position(bbox, self.dt, self.pixel_to_meter)
                active_objects.append(obj)
            else:
                obj_id = self.next_object_id
                self.next_object_id += 1
                new_obj = TrackedObject(obj_id, class_name, bbox)
                self.tracked_objects[obj_id] = new_obj
                active_objects.append(new_obj)

        new_near_misses = []
        for i in range(len(active_objects)):
            for j in range(i + 1, len(active_objects)):
                o1, o2 = active_objects[i], active_objects[j]
                ttc, dist, converging = self.calculate_ttc(o1, o2)
                
                vru = o1.is_vru or o2.is_vru
                risk = self.assess_risk(ttc, vru)
                
                if risk in ["CRITICAL", "HIGH RISK", "WARNING"]:
                    ev_id = f"NM-2026-{self.event_counter:04d}"
                    self.event_counter += 1
                    reason = "Trajectories Converging Rapidly" if risk == "CRITICAL" else "Close Following Separation"
                    ev = NearMissEvent(ev_id, o1, o2, ttc, risk, reason)
                    new_near_misses.append(ev)
                    self.near_miss_history.append(ev)

        return active_objects, new_near_misses

    def annotate_frame(self, frame: np.ndarray, tracked_objs: List[TrackedObject], near_misses: List[NearMissEvent]) -> np.ndarray:
        annotated = frame.copy()
        color_map = {
            "car": (255, 140, 0),
            "motorcycle": (0, 215, 255),
            "auto_rickshaw": (0, 255, 255),
            "bus": (255, 0, 128),
            "truck": (128, 0, 255),
            "pedestrian": (0, 255, 0)
        }

        for obj in tracked_objs:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            color = color_map.get(obj.class_name, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{obj.class_name.title()} #{obj.object_id} ({obj.speed_kmh:.0f}km/h)"
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + len(label) * 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if len(obj.history) > 1:
                pts = np.array(obj.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, color, 2)

        if near_misses:
            y_offset = 35
            for nm in near_misses[:3]:
                text = f"⚠️ {nm.risk_level}: {nm.interaction} | TTC: {nm.ttc}s"
                color = (0, 0, 255) if nm.risk_level == "CRITICAL" else (0, 140, 255)
                cv2.rectangle(annotated, (10, y_offset - 24), (540, y_offset + 6), (15, 15, 20), -1)
                cv2.rectangle(annotated, (10, y_offset - 24), (540, y_offset + 6), color, 2)
                cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 38

        return annotated


def generate_synthetic_indian_traffic_frame(frame_num: int, scenario_type: str = "normal") -> Tuple[np.ndarray, List[Dict]]:
    """
    Generates distinct, realistic synthetic Indian road scene frames for Scenarios 1 to 6.
    """
    h, w = 480, 854
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (35, 38, 42)
    
    # 4-lane Indian road markings
    lane_y1, lane_y2 = 120, 360
    cv2.rectangle(img, (0, lane_y1), (w, lane_y2), (55, 58, 62), -1)
    cv2.line(img, (0, lane_y1), (w, lane_y1), (0, 220, 255), 3)  # Yellow curb
    cv2.line(img, (0, lane_y2), (w, lane_y2), (255, 255, 255), 3) # White curb
    
    for x in range(0, w, 40):
        cv2.line(img, (x, 240), (x + 20, 240), (200, 200, 200), 2)
        
    t = (frame_num % 120) / 120.0
    
    if "Scenario 6" in scenario_type:
        # Non-lane-based wrong-side overtake / lane splitting
        car_x, car_y = int(500 - t * 300), 160
        bike_x, bike_y = int(120 + t * 420), int(170 - np.sin(t * np.pi) * 35)
        bus_x, bus_y = int(80 + t * 250), 200
        rick_x, rick_y = int(320 + t * 180), 280
        ped_x, ped_y = int(220), int(105)
        detected_boxes = [
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 90, car_y + 45)},
            {"class_id": 1, "bbox": (bike_x, bike_y, bike_x + 40, bike_y + 30)},
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 55, rick_y + 40)},
            {"class_id": 5, "bbox": (ped_x, ped_y, ped_x + 20, ped_y + 35)},
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 130, bus_y + 50)},
        ]
    elif "Scenario 5" in scenario_type:
        # Multi-Hotspot Emergency Alert (Heavy congested bottleneck + Bus/Truck queue)
        bus_x, bus_y = int(200 + t * 40), 160
        truck_x, truck_y = int(340 + t * 30), 165
        car_x, car_y = int(90 + t * 50), 230
        rick_x, rick_y = int(240 + t * 45), 280
        bike_x, bike_y = int(310 + t * 50), 235
        ped_x, ped_y = int(450), int(140 + np.sin(t * np.pi) * 80)
        detected_boxes = [
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 130, bus_y + 50)},
            {"class_id": 3, "bbox": (truck_x, truck_y, truck_x + 110, truck_y + 45)},
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 85, car_y + 40)},
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 55, rick_y + 40)},
            {"class_id": 1, "bbox": (bike_x, bike_y, bike_x + 40, bike_y + 30)},
            {"class_id": 5, "bbox": (ped_x, ped_y, ped_x + 20, ped_y + 35)},
        ]
    elif "Scenario 4" in scenario_type:
        # Dense Morning Rush Hour Bumper-to-Bumper Queue
        car_x, car_y = int(80 + t * 60), 150
        bike1_x, bike1_y = int(180 + t * 75), 145
        rick_x, rick_y = int(230 + t * 65), 220
        bus_x, bus_y = int(300 + t * 50), 210
        car2_x, car2_y = int(440 + t * 55), 280
        bike2_x, bike2_y = int(540 + t * 70), 285
        detected_boxes = [
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 85, car_y + 40)},
            {"class_id": 1, "bbox": (bike1_x, bike1_y, bike1_x + 35, bike1_y + 25)},
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 50, rick_y + 35)},
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 120, bus_y + 45)},
            {"class_id": 0, "bbox": (car2_x, car2_y, car2_x + 85, car2_y + 40)},
            {"class_id": 1, "bbox": (bike2_x, bike2_y, bike2_x + 35, bike2_y + 25)},
        ]
    elif "Scenario 3" in scenario_type:
        # Auto-Rickshaw & Bike Conflict
        rick_x, rick_y = int(220 + t * 280), 220
        bike_x, bike_y = int(rick_x - 30 + t * 40), int(210 + t * 20)
        car_x, car_y = int(60 + t * 400), 150
        bus_x, bus_y = int(450 + t * 150) % w, 290
        ped_x, ped_y = 650, 100
        detected_boxes = [
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 55, rick_y + 40)},
            {"class_id": 1, "bbox": (bike_x, bike_y, bike_x + 40, bike_y + 30)},
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 90, car_y + 45)},
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 130, bus_y + 50)},
            {"class_id": 5, "bbox": (ped_x, ped_y, ped_x + 20, ped_y + 35)},
        ]
    elif "Scenario 2" in scenario_type:
        # Pedestrian VRU Crossing Conflict
        car_x, car_y = int(140 + t * 500), 170
        bike_x, bike_y = int(car_x + 90), 165
        bus_x, bus_y = int(50 + t * 300), 300
        rick_x, rick_y = int(500 + t * 150) % w, 280
        ped_x, ped_y = int(360), int(110 + t * 160)
        detected_boxes = [
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 90, car_y + 45)},
            {"class_id": 1, "bbox": (bike_x, bike_y, bike_x + 40, bike_y + 30)},
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 55, rick_y + 40)},
            {"class_id": 5, "bbox": (ped_x, ped_y, ped_x + 20, ped_y + 35)},
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 130, bus_y + 50)},
        ]
    else:
        # Scenario 1: Normal Smooth Mixed Flow
        car_x, car_y = int(100 + t * 550), 170
        bike_x, bike_y = int(car_x + 180), 180
        rick_x, rick_y = int(520 + t * 200) % w, 280
        ped_x, ped_y = int(380 + np.sin(t * np.pi) * 15), int(100)
        bus_x, bus_y = int(50 + t * 400), 310
        detected_boxes = [
            {"class_id": 0, "bbox": (car_x, car_y, car_x + 90, car_y + 45)},
            {"class_id": 1, "bbox": (bike_x, bike_y, bike_x + 40, bike_y + 30)},
            {"class_id": 4, "bbox": (rick_x, rick_y, rick_x + 55, rick_y + 40)},
            {"class_id": 5, "bbox": (ped_x, ped_y, ped_x + 20, ped_y + 35)},
            {"class_id": 2, "bbox": (bus_x, bus_y, bus_x + 130, bus_y + 50)},
        ]
    
    scene_label = f"PR-VIGIL CAMERA STREAM [{scenario_type.upper()}]"
    cv2.putText(img, scene_label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img, detected_boxes
