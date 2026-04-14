
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
    slots = session.query(ParkingSlot).all()
    print(f"{'Slot ID':<15} | {'Available':<10} | {'Floor':<10}")
    print("-" * 40)
    for s in slots:
        print(f"{s.slot_id:<15} | {str(s.is_available):<10} | {s.floor:<10}")
finally:
    session.close()
