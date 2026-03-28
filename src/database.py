from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Module-level reference — set from main.py after initialization
_db_manager = None

def init_db(db_url: str):
    """Initialize the global DatabaseManager. Call once from main.py."""
    global _db_manager
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
