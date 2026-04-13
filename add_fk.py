import sqlalchemy
from src.database import init_db
from src.config import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    db_manager = init_db(config.database.url)
    
    with db_manager.engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("ALTER TABLE alerts ADD CONSTRAINT FK_alert_vehicle FOREIGN KEY (plate_number) REFERENCES vehicles(plate_number)"))
            conn.commit()
            print("Successfully added foreign key constraint from alerts.plate_number to vehicles.plate_number.")
        except Exception as e:
            print(f"Error altering table: {e}")
