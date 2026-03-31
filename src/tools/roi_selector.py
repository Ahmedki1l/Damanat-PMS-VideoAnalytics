import cv2
import json
import os
import yaml
import numpy as np
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_rtsp_url(camera_config, channel=102):
    user = camera_config["user"]
    password = camera_config["password"]
    ip = camera_config["ip"]
    return f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}"

class ROISelector:
    def __init__(self, camera_id, rtsp_url, json_path):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.json_path = json_path
        self.points = []
        self.window_name = f"ROI Selector - {camera_id} (Click to add points, 's' to save, 'c' to clear, 'q' to quit)"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.draw()

    def draw(self):
        img_copy = self.img.copy()
        if len(self.points) > 0:
            for pt in self.points:
                cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.polylines(img_copy, [np.array(self.points)], isClosed=False, color=(0, 255, 0), thickness=2)
            if len(self.points) > 2:
                # Preview closing the polygon
                cv2.line(img_copy, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 255), 1)
        
        cv2.imshow(self.window_name, img_copy)

    def run(self):
        print(f"[INFO] Connecting to {self.camera_id}...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url)
        
        ret, self.img = cap.read()
        cap.release()

        if not ret or self.img is None:
            print(f"[ERROR] Could not grab frame from {self.camera_id}. Check RTSP URL/Connection.")
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                self.draw()
            elif key == ord('s'):
                if len(self.points) >= 3:
                    self.save_roi()
                    break
                else:
                    print("[WARN] Need at least 3 points to save ROI.")

        cv2.destroyAllWindows()

    def save_roi(self):
        if not os.path.exists(self.json_path):
            data = []
        else:
            with open(self.json_path, "r") as f:
                data = json.load(f)

        # Remove existing ROI if any
        data = [entry for entry in data if entry["id"].lower() != "roi"]
        
        # Add new ROI
        data.append({
            "id": "roi",
            "polygon": self.points,
            "label": "Global ROI Mask"
        })

        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[SUCCESS] ROI saved to {self.json_path}")

if __name__ == "__main__":
    config = load_config()
    target_cams = ["CAM_01", "CAM_02"]
    
    print("Available Cameras for ROI definition:")
    for cam in config["cameras"]:
        if cam["id"] in target_cams:
            print(f" - {cam['id']}: {cam['name']} ({cam['floor']})")

    cam_id = input("\nEnter Camera ID to configure (or press Enter for CAM_01): ").strip() or "CAM_01"
    
    selected_cam = next((c for c in config["cameras"] if c["id"] == cam_id), None)
    if not selected_cam:
        print(f"[ERROR] Camera {cam_id} not found in config.")
    else:
        url = build_rtsp_url(selected_cam, channel=config["processing"].get("stream_channel", 102))
        selector = ROISelector(cam_id, url, selected_cam["slots_file"])
        selector.run()
