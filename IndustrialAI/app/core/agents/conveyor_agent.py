import cv2
import time
from ultralytics import YOLO
from ..base_agent import BaseAgent


class ConveyorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("Conveyor_Analytics", config)

        # Load the general-purpose model for objects (bottles, boxes, etc.)
        self.model = YOLO(self.config.get('model_path', 'models/yolo11n.pt'))

        self.line_y         = self.config.get('roi_line_y', 300)
        self.target_classes = self.config.get('target_classes', [39, 41, 28])

        # Counting state
        self.counted_ids     = set()
        self.total_count     = 0
        self.last_count_time = time.time()
        self.health_score    = 100.0

        # Active track memory for counted_ids pruning
        self.track_memory = {}
        self.TRACK_TTL    = 2.0   # seconds before a lost track is forgotten

    def reset(self):
        """Clears counters and tracking state (triggered by 'R' key)."""
        self.counted_ids.clear()
        self.track_memory.clear()
        self.total_count     = 0
        self.last_count_time = time.time()
        self.health_score    = 100.0
        self.log_event("MANUAL_RESET", {"status": "conveyor_cleared"})

    def process(self, frame):
        """
        Processes the raw frame using its own YOLO model.
        Counts objects that cross the ROI line and monitors conveyor health.
        """
        now = time.time()

        # 1. Run private inference
        results = self.model.track(
            frame,
            persist=True,
            conf=0.25,
            iou=0.5,
            imgsz=640,
            verbose=False
        )

        # 2. Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                if cls not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box
                cy = (y1 + y2) / 2

                # Update last-seen time for pruning
                self.track_memory[tid] = now

                # Line crossing logic (object moves top → bottom)
                if cy > self.line_y and tid not in self.counted_ids:
                    self.total_count += 1
                    self.counted_ids.add(tid)
                    self.last_count_time = now
                    self.log_event("OBJECT_COUNTED", {
                        "total": self.total_count,
                        "class": int(cls)
                    })

        # 3. Prune stale tracks — also remove from counted_ids to prevent memory leak
        for tid in list(self.track_memory.keys()):
            if now - self.track_memory[tid] > self.TRACK_TTL:
                del self.track_memory[tid]
                self.counted_ids.discard(tid)

        # 4. Health scoring (idle monitoring)
        time_since_last = now - self.last_count_time
        if time_since_last > 10.0:
            self.health_score = max(0.0, self.health_score - 0.1)
        else:
            self.health_score = min(100.0, self.health_score + 0.05)

        return {
            "count":  self.total_count,
            "health": round(self.health_score, 1),
            "status": "IDLE" if time_since_last > 5.0 else "ACTIVE",
        }

    def draw(self, frame, telemetry):
        # ROI counting line
        cv2.line(frame,
                 (0, self.line_y),
                 (frame.shape[1], self.line_y),
                 (255, 165, 0), 2)

        # HUD overlays
        cv2.putText(frame, f"PRODUCTION: {telemetry['status']}", (20, 100), 0, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Count: {telemetry['count']}",        (20, 130), 0, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Health: {telemetry['health']}%",     (20, 160), 0, 0.6, (0, 255, 0), 1)