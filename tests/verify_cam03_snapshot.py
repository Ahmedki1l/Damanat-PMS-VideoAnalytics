import os
import cv2
import numpy as np
import uuid
from datetime import datetime
from src.vehicle_registry import VehicleRegistry

def test_cam03_snapshot():
    registry = VehicleRegistry(image_dir="vehicle_images_test")
    
    # Clean up test dir if exists
    if os.path.exists("vehicle_images_test"):
        import shutil
        shutil.rmtree("vehicle_images_test")
    os.makedirs("vehicle_images_test", exist_ok=True)
    
    plate = "ABC-123"
    
    # 1. ANPR Entry
    registry.register_anpr_event(plate, "entry")
    
    # 2. CAM_01 Park Entry Candidate
    cam01_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(cam01_image, "CAM_01", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    candidate = registry.open_park_entry_candidate(camera_id="CAM_01", track_id=101)
    registry.update_park_entry_candidate_snapshot(candidate.candidate_id, cam01_image, quality_score=100.0)
    
    # Verify CAM_01 snapshot saved
    cand_files = [f for f in os.listdir("vehicle_images_test") if f.startswith(f"cand_")]
    assert len(cand_files) == 1, "CAM_01 snapshot should be saved"
    print(f"[OK] CAM_01 snapshot saved: {cand_files[0]}")
    
    # 3. Bind ANPR to Candidate
    registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
    
    # 4. CAM_03 Confirmation
    cam03_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(cam03_image, "CAM_03", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # We need to mock the matcher because matching happens before committing session
    # Actually, we can just ensure similarity is high since images are identical if we pass the same image,
    # but here they are different. We need to bypass the matcher or ensure it matches.
    
    # For testing, we can temporarily set the similarity_threshold to 0.0 or just use the same image.
    # But the registry uses reid_matcher and matcher.
    
    # Let's try to confirm with a very low threshold if possible, but the code has hardcoded 0.55/0.45.
    # Easiest way is to monkey-patch the similarity compute or just use the same image.
    
    # Let's monkey-patch for simplicity in this test
    registry.__dict__['_matcher'] = type('Mock', (), {'compare': lambda *args: 1.0, '_compare_dominant_colors': lambda *args: 1.0})()
    registry.__dict__['_reid_matcher'] = type('Mock', (), {'extract_feature': lambda *args: np.array([1.0]), 'compute_similarity': lambda *args: 1.0})()

    confirmed_plate = registry.confirm_at_b1_entrance(
        camera_id="CAM_03",
        track_id=202,
        image=cam03_image
    )
    
    assert confirmed_plate == plate, f"Expected {plate}, got {confirmed_plate}"
    print(f"[OK] Confirmed plate: {confirmed_plate}")
    
    # 5. Verify Session Snapshot
    # Find session for our plate
    session = None
    for s in registry._sessions.values():
        if s.plate == plate:
            session = s
            break
            
    assert session is not None, "Session should be found for confirmed plate"
    
    snapshot_path = session.snapshot_path
    print(f"Session snapshot path: {snapshot_path}")
    
    # Check disk for session file
    session_files = [f for f in os.listdir("vehicle_images_test") if f.startswith("session_")]
    assert len(session_files) == 1, "CAM_03 session snapshot should be saved"
    print(f"[OK] CAM_03 session snapshot saved: {session_files[0]}")
    
    # Verify session path matches the saved file
    assert session_files[0] in snapshot_path, f"Snapshot path {snapshot_path} should contain session file name {session_files[0]}"
    
    print("\n--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_cam03_snapshot()
