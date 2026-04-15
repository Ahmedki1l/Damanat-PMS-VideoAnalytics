
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("."))

from src.database import init_db
from src.model.parkingslot import ParkingSlot
from src.config import load_config

config = load_config("config.yaml")
db_manager = init_db(config.database.url)
session = db_manager.SessionLocal()

try:
    # Let's pick a slot that we know is physically empty but we want to mark as occupied in DB
    # Based on previous check, G3 was True. Let's make it False.
    slot_id = "G3"
    slot = session.query(ParkingSlot).filter(ParkingSlot.slot_id == slot_id).first()
    if slot:
        print(f"Setting {slot_id} to is_available=False (Simulating stale state)")
        slot.is_available = False
        session.commit()
    else:
        print(f"Slot {slot_id} not found.")
finally:
    session.close()
