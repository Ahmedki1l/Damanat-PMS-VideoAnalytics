import sqlalchemy
from src.database import get_db, _db_manager, init_db
from src.config import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    db_manager = init_db(config.database.url)
    
    with db_manager.engine.connect() as conn:
        try:
            conn.execute(sqlalchemy.text("ALTER TABLE alerts ADD plate_number VARCHAR(50) NULL"))
            conn.commit()
            print("Successfully added plate_number to alerts table.")
        except Exception as e:
            print(f"Error altering table: {e}")
