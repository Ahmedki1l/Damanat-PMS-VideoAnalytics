import os
from datetime import datetime
from src.model.alert import Alert

def test_alert_instantiation():
    try:
        # Create an alert instance with severity
        alert = Alert(
            alert_type="violation",
            camera_id="TEST_CAM",
            zone_id="TEST_ZONE",
            slot_id="TEST_SLOT",
            event_type="vehicle_detected",
            description="Test alert",
            is_resolved=False,
            triggered_at=datetime.utcnow(),
            severity="critical"
        )
        print("✓ Alert instantiated successfully with severity='critical'")
        print(f"  Alert Dict: {{'alert_type': {alert.alert_type}, 'severity': {alert.severity}}}")
        
        # Test default severity
        alert_default = Alert(
            alert_type="info",
            camera_id="TEST_CAM",
            event_type="test",
            is_resolved=True
        )
        print("✓ Alert instantiated successfully with default severity")
        print(f"  Default severity: {alert_default.severity}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate Alert: {e}")
        return False

if __name__ == "__main__":
    test_alert_instantiation()
