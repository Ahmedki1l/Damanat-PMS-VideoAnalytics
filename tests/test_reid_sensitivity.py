"""
test_reid_sensitivity.py — Calibration of ReID and Color thresholds.
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append('.')

from src.reid_matcher import get_reid_matcher
from src.image_matcher import VehicleImageMatcher

def run_test():
    reid_matcher = get_reid_matcher()
    legacy_matcher = VehicleImageMatcher()

    pairs = [
        # 1. Same car, same camera (NDD-4141)
        ("Same Cam/Same Car", 
         "vehicle_images/ERD-7800_CAM01_bound_20260413_073513.jpg", 
         "vehicle_images/ERD-7800_CAM01_bound_20260413_073518.jpg"),
        
        # 2. Same car, same camera (SDD-6707)
        ("Same Cam/Same Car 2",
         "vehicle_images/SDD-6707_CAM01_bound_20260413_073123.jpg",
         "vehicle_images/SDD-6707_CAM01_bound_20260413_073129.jpg"),

        # 3. Same car, different cameras (NDD-4141)
        ("Diff Cam/Same Car",
         "vehicle_images/NDD-4141_CAM01_bound_20260413_080606.jpg",
         "vehicle_images/NDD-4141_CAM_03_20260413_080621.jpg"),

        # 4. Different cars, same camera
        ("Same Cam/Diff Car",
         "vehicle_images/ERD-7800_CAM01_bound_20260413_073513.jpg",
         "vehicle_images/SDD-6707_CAM01_bound_20260413_073123.jpg")
    ]

    print("\n" + "="*80)
    print(f"{'TEST CASE':<25} | {'REID':<10} | {'COLOR':<10} | {'STATUS'}")
    print("="*80)

    for label, p1, p2 in pairs:
        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"{label:<25} | Missing Files")
            continue
            
        i1 = cv2.imread(p1)
        i2 = cv2.imread(p2)
        
        v1 = reid_matcher.extract_feature(i1)
        v2 = reid_matcher.extract_feature(i2)
        
        reid_score = reid_matcher.compute_similarity(v1, v2)
        color_score = legacy_matcher._compare_dominant_colors(i1, i2)
        
        status = "MATCH" if (reid_score > 0.4 and color_score > 0.4) else "MISS"
        print(f"{label:<25} | {reid_score:<10.3f} | {color_score:<10.3f} | {status}")

    print("="*80 + "\n")

if __name__ == "__main__":
    run_test()
