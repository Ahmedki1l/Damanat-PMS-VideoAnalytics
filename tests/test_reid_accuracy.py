"""
test_reid_accuracy.py - Verification of Phase 2 (Deep ReID Integration).

Tests the ReID matcher with real images of the same car across different cameras.
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append('.')

from src.reid_matcher import get_reid_matcher
from src.image_matcher import VehicleImageMatcher

def test_matching():
    # 1. Paths to sample images (Same car: NDD-4141)
    img1_path = "vehicle_images/NDD-4141_CAM01_bound_20260413_080606.jpg"
    img2_path = "vehicle_images/NDD-4141_CAM_03_20260413_080621.jpg"
    
    # 2. Path to a different car (e.g. SDD-6707)
    img_diff_path = "vehicle_images/SDD-6707_CAM01_bound_20260413_073123.jpg"

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Sample images not found in vehicle_images/. Run engine first?")
        return

    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img_diff = cv2.imread(img_diff_path)

    # Initialize matchers
    reid_matcher = get_reid_matcher()
    legacy_matcher = VehicleImageMatcher()

    print("\n" + "="*60)
    print("  PHASE 2: RE-ID ACCURACY TEST")
    print("="*60)

    # --- Test Case 1: Same Car (Cross-Camera) ---
    print("\n[MATCH] Same Car (CAM_01 vs CAM_03):")
    feat1 = reid_matcher.extract_feature(img1)
    feat2 = reid_matcher.extract_feature(img2)
    
    score = reid_matcher.compute_similarity(feat1, feat2)
    color_score = legacy_matcher._compare_dominant_colors(img1, img2)
    
    print(f"  -> ReID Similarity: {score:.3f}")
    print(f"  -> Color Similarity: {color_score:.3f}")
    print(f"  -> PASS? {'YES' if score > 0.35 and color_score > 0.6 else 'NO'}")

    # --- Test Case 2: Different Car ---
    print("\n[MISMATCH] Different Cars (CAM_01 vs CAM_01):")
    feat_diff = reid_matcher.extract_feature(img_diff)
    
    score_diff = reid_matcher.compute_similarity(feat1, feat_diff)
    color_score_diff = legacy_matcher._compare_dominant_colors(img1, img_diff)
    
    print(f"  -> ReID Similarity: {score_diff:.3f}")
    print(f"  -> Color Similarity: {color_score_diff:.3f}")
    print(f"  -> CORRECTLY REJECTED? {'YES' if score_diff < 0.35 or color_score_diff < 0.6 else 'NO'}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_matching()
