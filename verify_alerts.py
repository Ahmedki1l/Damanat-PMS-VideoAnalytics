
import time
import numpy as np
import cv2
from src.models.state_machine import SlotStateMachine, SlotEvent
from src.core.engine import ParkingEngine
from src.config import load_config
from datetime import datetime

def test_logic():
    print("--- Testing Parking Violation Logic ---")
    
    # 1. Test State Machine Property
    sm = SlotStateMachine(slot_id="Violation_Zone")
    assert sm.is_violation_zone == True
    print("[OK] State Machine identifies violation zones correctly.")

    # 2. Test Engine Logic (Simulated)
    from src.vehicle_registry import VehicleRegistry
    config = load_config("config.yaml")
    registry = VehicleRegistry(image_dir="/tmp/test_images")
    engine = ParkingEngine(config, vehicle_registry=registry)
    
    # Mock frame and detection
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    class MockDet:
        bbox = [10, 10, 50, 50]
    
    # First violation event
    evt = SlotEvent(
        event_type="vehicle_parked",
        slot_id="Violation",
        track_id=1,
        timestamp=datetime.now().isoformat()
    )
    
    print("\nSimulating first violation...")
    # We manually trigger the logic part from engine.py (simplified for test)
    # In real run, engine.run_multi_camera does this
    
    # Helper to simulate engine's event processing
    def process_evt(eng, event, crop):
        now_ts = time.time()
        # Clean up old
        eng._recent_violators = [v for v in eng._recent_violators if now_ts - v['timestamp'] < eng._violation_history_limit]
        
        # Check similarity
        is_duplicate = False
        for v in eng._recent_violators:
            score = eng.vehicle_registry.matcher.compare(crop, v['crop'])
            if score > eng._violation_match_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            event.event_type = "vehicle_violation"
            eng._recent_violators.append({'crop': crop.copy(), 'timestamp': now_ts})
            return True # Alert sent
        return False # Suppressed
    
    # Create two different "cars" (different colors)
    car_white = np.ones((40, 40, 3), dtype=np.uint8) * 255
    car_black = np.zeros((40, 40, 3), dtype=np.uint8)
    
    # Step A: First car arrives -> Alert
    alerted = process_evt(engine, evt, car_white)
    print(f"  First car (White): Alert sent? {alerted}")
    assert alerted == True
    assert evt.event_type == "vehicle_violation"

    # Step B: Same car seen again (Simulating another camera) -> No Alert
    evt2 = SlotEvent(event_type="vehicle_parked", slot_id="Violation_2", track_id=2, timestamp=datetime.now().isoformat())
    alerted = process_evt(engine, evt2, car_white)
    print(f"  Same car (White) again: Alert sent? {alerted} (Should be False)")
    assert alerted == False

    # Step C: Different car arrives -> Alert
    evt3 = SlotEvent(event_type="vehicle_parked", slot_id="Violation", track_id=3, timestamp=datetime.now().isoformat())
    alerted = process_evt(engine, evt3, car_black)
    print(f"  Second car (Black) arriving simultaneously: Alert sent? {alerted} (Should be True)")
    assert alerted == True
    assert evt3.event_type == "vehicle_violation"

    print("\n--- ALL LOGIC TESTS PASSED ---")

if __name__ == "__main__":
    test_logic()
