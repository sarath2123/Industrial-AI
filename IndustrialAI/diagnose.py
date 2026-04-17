"""
diagnose_ppe.py
───────────────
Standalone script to verify your best.pt model is detecting correctly.
Run this BEFORE the full SentinelAI stack to isolate detection issues.

Usage:
    python diagnose_ppe.py                   # uses webcam 0
    python diagnose_ppe.py --source video.mp4
    python diagnose_ppe.py --model path/to/best.pt
"""

import cv2
import time
import argparse
from ultralytics import YOLO

# ── Model class map (from best.pt) ──────────────────────────────────────────
CLASS_NAMES = {
    0:  "Fall-Detected",
    1:  "Gloves",
    2:  "Goggles",
    3:  "Hardhat",
    4:  "Ladder",
    5:  "Mask",
    6:  "NO-Gloves",
    7:  "NO-Goggles",
    8:  "NO-Hardhat",
    9:  "NO-Mask",
    10: "NO-Safety Vest",
    11: "Person",
    12: "Safety Cone",
    13: "Safety Vest",
}

VIOLATION_CLASSES = {
    "NO-Hardhat", "NO-Safety Vest", "NO-Gloves",
    "NO-Goggles", "NO-Mask", "Fall-Detected"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,         help="Camera index or video path")
    parser.add_argument("--model",  default="models/best.pt", help="Path to best.pt")
    parser.add_argument("--conf",   type=float, default=0.25,  help="Confidence threshold")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Verify class names match expectations
    print("\nModel class map:")
    for idx, name in model.names.items():
        expected = CLASS_NAMES.get(idx, "?")
        match = "OK" if name == expected else f"MISMATCH — expected '{expected}'"
        print(f"  {idx:2d}: {name:<20} {match}")
    print()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open source: {args.source}")
        return

    print("Diagnostic running — press 'q' to quit, 'r' to print detection log")
    print(f"Confidence threshold: {args.conf}")
    print()

    detection_log   = []
    prev_time       = time.time()
    frame_count     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        frame = cv2.flip(frame, 1)  # mirror for webcam
        frame_count += 1

        # Run inference (no tracking in diagnostic — simpler)
        results = model.predict(frame, conf=args.conf, iou=0.5, imgsz=640, verbose=False)

        detections_this_frame = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls]
                detections_this_frame.append(f"{label} ({conf:.0%})")

                is_violation = label in VIOLATION_CLASSES
                color = (0, 0, 255) if is_violation else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.0%}",
                            (x1, max(y1 - 6, 10)), 0, 0.5, color, 1)

        # FPS
        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_count}",
                    (10, 25), 0, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Detections: {len(detections_this_frame)}",
                    (10, 50), 0, 0.6, (255, 255, 0), 1)

        if detections_this_frame:
            detection_log.append((frame_count, detections_this_frame))
            # Print to terminal every detection
            print(f"[Frame {frame_count:05d}] {', '.join(detections_this_frame)}")

        cv2.imshow("PPE Diagnostic — best.pt", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            print(f"\n--- Detection log ({len(detection_log)} frames with detections) ---")
            for f, dets in detection_log[-20:]:
                print(f"  Frame {f}: {', '.join(dets)}")
            print()

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total frames processed : {frame_count}")
    print(f"Frames with detections : {len(detection_log)}")

    if detection_log:
        all_labels = [d for _, dets in detection_log for d in dets]
        from collections import Counter
        counts = Counter(d.split(' ')[0] for d in all_labels)
        print("Class counts:")
        for label, count in counts.most_common():
            print(f"  {label:<20} {count}")
    else:
        print("WARNING: Zero detections in entire run.")
        print("Check:")
        print("  1. Is models/best.pt the correct path?")
        print("  2. Is the camera showing you clearly (good lighting)?")
        print("  3. Try lowering --conf to 0.10")
        print("  4. Are you inside the camera frame?")


if __name__ == "__main__":
    main()