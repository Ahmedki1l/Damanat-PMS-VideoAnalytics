"""
Utility script to create config and preprocessing_config tables in the database.
Uses the existing SQLAlchemy models defined in the codebase.
"""

import os
import sys

# Add src to python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.config import load_config
from src.database import init_db
from src.model.config_run import Config, PreprocessingConfig

def main():
    # 1. Load configuration to get DATABASE_URL
    print("[INFO] Loading configuration...")
    app_config = load_config("config.yaml")
    
    db_url = app_config.database.url
    if not db_url:
        print("[ERROR] DATABASE_URL not found in config.yaml under 'database.DATABASE_URL'")
        print("Please ensure your config.yaml is set up correctly.")
        sys.exit(1)
        
    print(f"[INFO] Initializing database at: {db_url}")
    
    # 2. Initialize database and create tables
    try:
        db_manager = init_db(db_url)
        print("[INFO] Creating tables (config, preprocessing_config)...")
        
        # This will create all tables defined with 'Base' that don't exist yet
        db_manager.create_tables()
        
        print("[SUCCESS] Tables created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to create tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
