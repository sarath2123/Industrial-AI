import cv2
import time
from ultralytics import YOLO
from ..base_agent import BaseAgent

# ── Exact class names from best.pt ──────────────────────────────────────────
#  0: Fall-Detected     1: Gloves           2: Goggles
#  3: Hardhat           4: Ladder           5: Mask
#  6: NO-Gloves         7: NO-Goggles       8: NO-Hardhat
#  9: NO-Mask          10: NO-Safety Vest  11: Person
# 12: Safety Cone      13: Safety Vest
# ─────────────────────────────────────────────────────────────────────────────

VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Mask",
    "Fall-Detected",
}

COMPLIANT_CLASSES = {
    "Hardhat",
    "Safety Vest",
    "Gloves",
    "Goggles",
    "Mask",
}

# Risk points added per active violation type
VIOLATION_RISK = {
    "NO-Hardhat":      30,
    "NO-Safety Vest":  25,
    "NO-Gloves":       15,
    "NO-Goggles":      10,
    "NO-Mask":         10,
    "Fall-Detected":   50,
}

# BGR colors for bounding box labels
LABEL_COLORS = {
    "NO-Hardhat":     (0,   0,   255),
    "NO-Safety Vest": (0,   0,   255),
    "NO-Gloves":      (0,   80,  255),
    "NO-Goggles":     (0,   80,  255),
    "NO-Mask":        (0,   80,  255),
    "Fall-Detected":  (0,   0,   180),
    "Hardhat":        (0,   255, 0  ),
    "Safety Vest":    (0,   255, 0  ),
    "Gloves":         (0,   200, 0  ),
    "Goggles":        (0,   200, 0  ),
    "Mask":           (0,   200, 0  ),
    "Person":         (255, 255, 0  ),
    "Safety Cone":    (0,   165, 255),
    "Ladder":         (200, 200, 200),
}


class PPEAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("PPE_Safety", config)

        self.model = YOLO(self.config.get('model_path', 'models/best.pt'))

        self.zone = self.config.get('zone', [400, 100, 620, 400])

        # Risk thresholds from settings.yaml
        self.RISK_CAUTION = self.config.get('risk_thresholds', {}).get('caution', 25)
        self.RISK_WARNING = self.config.get('risk_thresholds', {}).get('warning', 45)
        self.RISK_STOP    = self.config.get('risk_thresholds', {}).get('stop',    65)

        # Hysteresis: frames to confirm / clear a violation
        self.VIOLATION_ON  = 3
        self.VIOLATION_OFF = 6

        # Seconds to keep a lost track in memory
        self.TRACK_TTL = 1.5

        # State
        self.track_memory    = {}
        self.is_latched      = False
        self.risk_hold_start = None

    def reset(self):
        """Clears latch and all tracking state (triggered by 'R' key)."""
        self.is_latched      = False
        self.risk_hold_start = None
        self.track_memory.clear()
        self.log_event("MANUAL_RESET", {"status": "system_cleared"})

    def process(self, frame):
        now        = time.time()
        frame_risk = 0

        # ── 1. Inference ──────────────────────────────────────────────────────
        results = self.model.track(
            frame,
            persist=True,
            conf=0.10,
            iou=0.5,
            imgsz=640,
            verbose=False,
        )

        # ── 2. Per-detection loop ─────────────────────────────────────────────
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            names = results[0].names   # {0: 'Fall-Detected', 1: 'Gloves', ...}

            for box, cls, track_id, conf in zip(boxes, clss, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                label   = names[cls]   # exact string e.g. "NO-Hardhat", "Safety Vest"
                in_zone = (self.zone[0] <= cx <= self.zone[2] and
                           self.zone[1] <= cy <= self.zone[3])

                # ── Init track entry ──────────────────────────────────────────
                if track_id not in self.track_memory:
                    self.track_memory[track_id] = {
                        "violations": {},  # label -> {on, off, active}
                        "last_seen":  now,
                    }

                t = self.track_memory[track_id]
                t["last_seen"] = now

                # ── Violation state machine ───────────────────────────────────
                if in_zone:
                    if label in VIOLATION_CLASSES:
                        v = t["violations"].setdefault(
                            label, {"on": 0, "off": 0, "active": False}
                        )
                        v["on"]  += 1
                        v["off"]  = 0
                        if v["on"] >= self.VIOLATION_ON and not v["active"]:
                            v["active"] = True
                            self.log_event(
                                f"{label.upper().replace(' ', '_').replace('-', '_')}_VIOLATION",
                                {"track": int(track_id), "conf": round(float(conf), 2)},
                            )

                    if label in COMPLIANT_CLASSES:
                        # Clear the matching violation counter
                        paired = f"NO-{label}"
                        if paired in t["violations"]:
                            v = t["violations"][paired]
                            v["off"] += 1
                            v["on"]   = 0
                            if v["off"] >= self.VIOLATION_OFF:
                                v["active"] = False

                # ── Draw bounding box with label ──────────────────────────────
                color = LABEL_COLORS.get(label, (200, 200, 200))
                if not in_zone:
                    # Dim detections outside the danger zone
                    color = tuple(max(0, c - 100) for c in color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.0%}",
                    (x1, max(y1 - 6, 10)), 0, 0.45, color, 1,
                )

        # ── 3. Prune stale tracks ─────────────────────────────────────────────
        for tid in list(self.track_memory.keys()):
            if now - self.track_memory[tid]["last_seen"] > self.TRACK_TTL:
                del self.track_memory[tid]

        # ── 4. Compute frame risk ─────────────────────────────────────────────
        person_count      = len(self.track_memory)
        active_violations = set()

        for t in self.track_memory.values():
            for label, v in t["violations"].items():
                if v["active"]:
                    active_violations.add(label)
                    frame_risk += VIOLATION_RISK.get(label, 20)

        if person_count > 1:
            frame_risk += (person_count - 1) * 5

        frame_risk = min(frame_risk, 100)

        # ── 5. Escalation timer ───────────────────────────────────────────────
        if frame_risk >= self.RISK_CAUTION:
            if self.risk_hold_start is None:
                self.risk_hold_start = now
            elapsed = now - self.risk_hold_start
            if elapsed > 5:
                frame_risk = max(frame_risk, self.RISK_STOP)
            elif elapsed > 3:
                frame_risk = max(frame_risk, self.RISK_WARNING)
        else:
            if not self.is_latched:
                self.risk_hold_start = None

        # ── 6. Latch ──────────────────────────────────────────────────────────
        if frame_risk >= self.RISK_STOP and not self.is_latched:
            self.is_latched = True
            self.log_event("PPE_STOP_TRIGGERED", {
                "risk":       frame_risk,
                "violations": list(active_violations),
            })

        if self.is_latched:
            status = "STOPPED"
        elif frame_risk >= self.RISK_WARNING:
            status = "WARNING"
        elif frame_risk >= self.RISK_CAUTION:
            status = "CAUTION"
        else:
            status = "SAFE"

        return {
            "risk_level":        frame_risk,
            "is_latched":        self.is_latched,
            "status":            status,
            "active_violations": list(active_violations),
        }

    def draw(self, frame, telemetry):
        if telemetry['is_latched']:
            color = (0, 0, 255)
        elif telemetry['risk_level'] >= self.RISK_WARNING:
            color = (0, 165, 255)
        elif telemetry['risk_level'] >= self.RISK_CAUTION:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        # Danger zone border
        cv2.rectangle(frame,
                      (self.zone[0], self.zone[1]),
                      (self.zone[2], self.zone[3]),
                      color, 2)

        cv2.putText(frame, f"SAFETY: {telemetry['status']}",    (20, 40),  0, 0.7, color, 2)
        cv2.putText(frame, f"Risk: {telemetry['risk_level']}%", (20, 70),  0, 0.6, (255, 255, 255), 1)

        # Active violation list on HUD
        y = 100
        for v in telemetry.get('active_violations', []):
            cv2.putText(frame, f"  ! {v}", (20, y), 0, 0.5, (0, 0, 255), 1)
            y += 22

        if telemetry['is_latched']:
            cv2.putText(frame, "LATCHED - PRESS 'R' TO RESET",
                        (20, y + 10), 0, 0.6, (0, 0, 255), 2)