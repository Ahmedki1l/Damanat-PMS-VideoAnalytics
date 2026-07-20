from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import make_url

Base = declarative_base()

# Covering index for SlotStatusRepository.get_latest_by_slot. Named so the
# back-fill in _ensure_schema_updates can detect it on an existing database —
# this repo has no migrations, so create_all alone will not add it to a table
# that already exists.
_SLOT_STATUS_LOOKUP_INDEX = "ix_slot_status_slot_id_time"

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

    def ping(self) -> bool:
        """Is the database actually reachable RIGHT NOW?

        ``db_manager is not None`` only proves the object was constructed — the server
        can be down while it stays truthy. The health check needs a live signal, so run
        the cheapest possible round-trip (``SELECT 1``) and report whether it answered.
        Any failure (connection refused, timeout, auth) is swallowed into ``False`` —
        this is a health probe, never a place to raise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

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
            with self.engine.begin() as conn:
                if "camera_id" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD camera_id VARCHAR(50) NULL"
                        )
                    )
                if "slot_type" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD slot_type VARCHAR(30) NOT NULL "
                            "CONSTRAINT DF_parking_slots_slot_type DEFAULT 'parking'"
                        )
                    )
                if "zone_id" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD zone_id VARCHAR(100) NULL"
                        )
                    )
                if "zone_name" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD zone_name VARCHAR(100) NULL"
                        )
                    )
                if "last_snapshot_path" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD last_snapshot_path VARCHAR(255) NULL"
                        )
                    )
                if "reservation_type" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD reservation_type VARCHAR(20) NOT NULL "
                            "CONSTRAINT DF_parking_slots_reservation_type DEFAULT 'GENERAL'"
                        )
                    )
                if "reserved_for" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE parking_slots "
                            "ADD reserved_for VARCHAR(255) NULL"
                        )
                    )

                # ---- Plate-lock binding columns (dialect-aware) ----------------
                # Production runs on MSSQL (BIT + named-constraint defaults); the
                # test suite runs on SQLite (BOOLEAN/no named constraint). Branch
                # so both create_all and this back-fill succeed on either engine.
                dialect = self.engine.dialect.name
                if dialect == "mssql":
                    bool_type, bool_def = "BIT", "0"
                    add_kw = "ADD"

                    def _default(name, val):
                        return f"CONSTRAINT DF_parking_slots_{name} DEFAULT {val}"
                else:  # sqlite (tests) and other engines
                    bool_type, bool_def = "BOOLEAN", "0"
                    add_kw = "ADD COLUMN"

                    def _default(name, val):
                        return f"DEFAULT {val}"

                # Nullable columns omit an explicit NULL keyword — nullable is the
                # default in both dialects, and SQLite's ADD COLUMN rejects a bare
                # NULL constraint (only MSSQL accepts it).
                if "current_plate" not in columns:
                    conn.execute(text(
                        f"ALTER TABLE parking_slots {add_kw} current_plate VARCHAR(50)"
                    ))
                if "plate_confidence" not in columns:
                    conn.execute(text(
                        f"ALTER TABLE parking_slots {add_kw} plate_confidence FLOAT "
                        f"NOT NULL {_default('plate_confidence', '0')}"
                    ))
                if "plate_locked" not in columns:
                    conn.execute(text(
                        f"ALTER TABLE parking_slots {add_kw} plate_locked {bool_type} "
                        f"NOT NULL {_default('plate_locked', bool_def)}"
                    ))
                if "plate_locked_at" not in columns:
                    conn.execute(text(
                        f"ALTER TABLE parking_slots {add_kw} plate_locked_at DATETIME"
                    ))

        # ---- slot_status lookup index (occupancy publish latency) ----------- #
        # SlotStatusRepository.get_latest_by_slot runs
        #     WHERE slot_id = ? ORDER BY time DESC
        # and BOTH columns were unindexed, on a table that is append-only and has
        # no retention — so it degrades into a full scan + sort as the table grows.
        #
        # Why it matters here specifically: that query runs inside
        # log_vehicle_event on every VACATE (the occupied->empty check) and on
        # every plate resolution, on the engine's consumer thread — and the
        # db.commit() that publishes `parking_slots.is_available` to the frontend
        # happens AFTER it. So the scan sits directly between a slot going vacant
        # and the UI being able to see it, and gets slower every day the system
        # runs. Adding the index is the fix; pruning old rows is the follow-up.
        if "slot_status" in table_names:
            try:
                existing = {ix["name"] for ix in inspector.get_indexes("slot_status")}
            except Exception:
                existing = set()
            if _SLOT_STATUS_LOOKUP_INDEX not in existing:
                try:
                    with self.engine.begin() as conn:
                        conn.execute(text(
                            f"CREATE INDEX {_SLOT_STATUS_LOOKUP_INDEX} "
                            "ON slot_status (slot_id, time DESC)"
                        ))
                    print(
                        f"[DB] created {_SLOT_STATUS_LOOKUP_INDEX} on "
                        "slot_status(slot_id, time DESC)"
                    )
                except Exception as exc:
                    # Never block startup on an index: the system is correct
                    # without it, just slower.
                    print(
                        f"[WARN] could not create {_SLOT_STATUS_LOOKUP_INDEX}: {exc}"
                    )
