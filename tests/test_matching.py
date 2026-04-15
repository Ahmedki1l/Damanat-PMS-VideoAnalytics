"""
test_matching.py — Standalone test for vehicle image matching.

Loads the latest gate crop image, grabs a live frame from a B1 camera,
detects cars, and runs the image matcher to see per-feature scores.

Usage:
    python test_matching.py
"""

import os
import glob
import cv2
import numpy as np
from src.image_matcher import VehicleImageMatcher


# --- Config ---
GATE_IMAGE = None  # Auto-find latest, or set manually e.g. "vehicle_images/ZUR-8870_gate.jpg"
TEST_PLATE = "ZUR-1111"

# B1 camera RTSP (CAM_04 - parking view)
CAM_IP = "10.1.13.63"
CAM_USER = "kloudspot"
CAM_PASS = "Kloud@123"
CAM_CHANNEL = 102  # sub-stream 720p
RTSP_URL = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/{CAM_CHANNEL}"


def find_latest_gate_crop():
    """Find the most recent gate crop image."""
    patterns = [
        "vehicle_images/*_gate.jpg",
        "vehicle_images/*.jpg",
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            return latest
    return None


def detect_cars_simple(frame):
    """Simple car detection using YOLO."""
    from ultralytics import YOLO

    model = YOLO("models/yolo11n.pt")
    results = model(frame, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # Class 2 = car, 5 = bus, 7 = truck
            if cls in (2, 5, 7):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

    return boxes


def main():
    print("=" * 60)
    print("  VEHICLE IMAGE MATCHING TEST")
    print("=" * 60)

    # --- 1. Load gate crop ---
    gate_path = GATE_IMAGE or find_latest_gate_crop()
    if not gate_path or not os.path.exists(gate_path):
        print(f"\n[ERROR] No gate crop image found!")
        print(f"  Looked in: vehicle_images/")
        print(f"  Run the system first to capture a gate crop.")
        return

    gate_img = cv2.imread(gate_path)
    if gate_img is None:
        print(f"[ERROR] Failed to read: {gate_path}")
        return

    print(f"\n[1] Gate crop loaded: {gate_path}")
    print(f"    Size: {gate_img.shape[1]}x{gate_img.shape[0]}")

    # Show the gate crop
    cv2.imshow("Gate Crop (Reference)", gate_img)
    cv2.waitKey(1)

    # --- 2. Grab frame from B1 camera ---
    print(f"\n[2] Connecting to B1 camera: {CAM_IP}...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[ERROR] Cannot connect to {RTSP_URL}")
        return

    # Read a few frames to stabilize
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to grab frame")
        return

    print(f"    Frame grabbed: {frame.shape[1]}x{frame.shape[0]}")

    # --- 3. Detect cars ---
    print(f"\n[3] Detecting cars...")
    boxes = detect_cars_simple(frame)
    print(f"    Found {len(boxes)} car(s)")

    if not boxes:
        print("[INFO] No cars detected in frame. Try a different camera.")
        cv2.imshow("B1 Camera Frame", frame)
        cv2.waitKey(0)
        return

    # --- 4. Run matching ---
    matcher = VehicleImageMatcher()
    print(f"\n[4] Running image matching (gate crop vs each detected car):")
    print("-" * 60)

    best_score = 0.0
    best_idx = -1
    annotated = frame.copy()

    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        # Crop the car
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0:
            continue

        print(f"\n  Car #{i} (conf={conf:.2f}, box=[{x1},{y1},{x2},{y2}]):")

        # Run matching
        score = matcher.compare(gate_img, car_crop)
        print(f"  → FINAL SCORE: {score:.3f} ({score:.1%})")

        # Track best
        if score > best_score:
            best_score = score
            best_idx = i

        # Draw box on frame
        color = (0, 255, 0) if score >= 0.35 else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"Car#{i} score={score:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show the car crop
        crop_resized = cv2.resize(car_crop, (200, 150))
        cv2.imshow(f"Car #{i} (score={score:.2f})", crop_resized)

    # --- 5. Results ---
    print("\n" + "=" * 60)
    if best_idx >= 0 and best_score >= 0.35:
        print(f"  ✓ MATCH FOUND! Car #{best_idx} → Plate: {TEST_PLATE}")
        print(f"    Score: {best_score:.3f} ({best_score:.1%})")

        # Label the matched car
        bx1, by1, bx2, by2, _ = boxes[best_idx]
        cv2.putText(annotated, f"PLATE: {TEST_PLATE}", (bx1, by2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        print(f"  ✗ NO MATCH (best score: {best_score:.3f}, threshold: 0.35)")
    print("=" * 60)

    cv2.imshow("B1 Camera - Matching Results", annotated)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
