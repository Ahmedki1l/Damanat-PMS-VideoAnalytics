from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import make_url

Base = declarative_base()

# Module-level reference — set from main.py after initialization
_db_manager = None

def init_db(db_url: str):
    """Initialize the global DatabaseManager. Call once from main.py."""
    global _db_manager
    
    # Auto-create database if it doesn't exist (handle SQL Server specifically)
    url_obj = make_url(db_url)
    db_name = url_obj.database
    
    if url_obj.get_dialect().name == 'mssql' and db_name:
        master_url = url_obj.set(database='master')
        engine = create_engine(master_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT name FROM sys.databases WHERE name = '{db_name}'")).fetchone()
                if not result:
                    print(f"[DB] Creating database '{db_name}' automatically...")
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
        except Exception as e:
            print(f"[DB WARN] Could not auto-create database (might already exist or lack permissions): {e}")
        finally:
            engine.dispose()

    _db_manager = DatabaseManager(db_url)
    return _db_manager

def get_db():
    """FastAPI dependency that yields a DB session."""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    db = _db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
        self._ensure_schema_updates()

    def _ensure_schema_updates(self):
        inspector = inspect(self.engine)
        try:
            table_names = set(inspector.get_table_names())
        except Exception:
            return

        if "parking_slots" in table_names:
            columns = {column["name"] for column in inspector.get_columns("parking_slots")}
            if "last_snapshot_path" not in columns:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD last_snapshot_path VARCHAR(255) NULL"
                        )
                    )
