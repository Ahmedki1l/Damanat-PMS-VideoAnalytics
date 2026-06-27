from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from src.model.config_run import Config, PreprocessingConfig
from src.model.parking_area import ParkingArea
from src.schemas import ScopeEnum
from src.config import AppConfig, AreaEntry, CameraEntry
from src.utils.crypto import make_decrypt

def ensure_config_initialized(db: Session, app_config: AppConfig):
    """
    Ensure that the Config table has at least one row.
    If empty, it seeds it using values from config.yaml.
    """
    existing = db.query(Config).first()
    if existing:
        return existing

    print("[DB] Initializing default configuration in database...")
    
    # 1. Create main Config row
    # Mapping AppConfig (from YAML) to Config (DB model)
    new_config = Config(
        target_fps=app_config.processing.target_fps_per_camera,
        processing_mode=app_config.processing.mode,
        stream_channel=app_config.processing.stream_channel,
        slot_ref_width=app_config.processing.slot_ref_width,
        slot_ref_height=app_config.processing.slot_ref_height,
        
        device=app_config.detector.device,
        model_path=app_config.detector.model_path,
        confidence=app_config.detector.confidence,
        imgsz=app_config.detector.imgsz,
        classes=",".join(map(str, app_config.detector.classes)),
        
        tracker_type=app_config.tracker.type,
        
        confirm_enter_frames=app_config.state_machine.confirm_enter_frames,
        confirm_leave_frames=app_config.state_machine.confirm_leave_frames,
        
        overlap_threshold=app_config.assigner.overlap_threshold,
        
        show_video=app_config.output.show_video,
        show_camera=app_config.output.show_camera,
        log_file=app_config.output.log_file
    )
    db.add(new_config)
    db.flush()  # To get new_config.id

    # 2. Create Preprocessing entries
    # Detector Preprocessing
    det_pp = PreprocessingConfig(
        config_id=new_config.id,
        scope=ScopeEnum.DETECTOR,
        enabled=app_config.preprocessing.detector.enabled,
        clip_limit=app_config.preprocessing.detector.clip_limit,
        grid_w=app_config.preprocessing.detector.grid_size[0],
        grid_h=app_config.preprocessing.detector.grid_size[1],
        gamma_correction=app_config.preprocessing.detector.gamma_correction,
        dark_threshold=app_config.preprocessing.detector.dark_threshold
    )
    db.add(det_pp)

    # ReID Preprocessing
    reid_pp = PreprocessingConfig(
        config_id=new_config.id,
        scope=ScopeEnum.REID,
        enabled=app_config.preprocessing.reid.enabled,
        clip_limit=app_config.preprocessing.reid.clip_limit,
        grid_w=app_config.preprocessing.reid.grid_size[0],
        grid_h=app_config.preprocessing.reid.grid_size[1]
    )
    db.add(reid_pp)

    db.commit()
    print("[DB] Default configuration initialized successfully.")
    return new_config


def sync_app_config_from_db(db: Session, app_config: AppConfig):
    """
    Load settings from the database and override the current app_config.
    This makes the database the source of truth for runtime settings.
    """
    db_config = db.query(Config).first()
    if not db_config:
        print("[WARN] No configuration found in database. Using YAML defaults.")
        return app_config

    print("[DB] Linking database configuration to runtime...")

    # 1. Processing settings
    app_config.processing.target_fps_per_camera = db_config.target_fps
    app_config.processing.mode = db_config.processing_mode.value if hasattr(db_config.processing_mode, 'value') else str(db_config.processing_mode)
    app_config.processing.stream_channel = db_config.stream_channel.value if hasattr(db_config.stream_channel, 'value') else int(db_config.stream_channel)
    app_config.processing.slot_ref_width = db_config.slot_ref_width
    app_config.processing.slot_ref_height = db_config.slot_ref_height

    # 2. Detector settings
    app_config.detector.device = db_config.device.value if hasattr(db_config.device, 'value') else str(db_config.device)
    app_config.detector.model_path = db_config.model_path
    app_config.detector.confidence = db_config.confidence
    app_config.detector.imgsz = db_config.imgsz
    if db_config.classes:
        try:
            app_config.detector.classes = [int(c.strip()) for c in db_config.classes.split(",") if c.strip()]
        except ValueError:
            print(f"[WARN] Invalid classes in DB: '{db_config.classes}'. Keeping YAML defaults.")

    # 3. Tracker settings
    app_config.tracker.type = db_config.tracker_type.value if hasattr(db_config.tracker_type, 'value') else str(db_config.tracker_type)

    # 4. State Machine & Assigner
    app_config.state_machine.confirm_enter_frames = db_config.confirm_enter_frames
    app_config.state_machine.confirm_leave_frames = db_config.confirm_leave_frames
    app_config.assigner.overlap_threshold = db_config.overlap_threshold

    # 5. Output settings
    app_config.output.show_video = db_config.show_video
    app_config.output.show_camera = db_config.show_camera
    app_config.output.log_file = db_config.log_file

    # 6. Preprocessing (linked rows)
    for pp in db_config.preprocessing:
        # Check if Enum matches scope
        scope_val = pp.scope.value if hasattr(pp.scope, 'value') else str(pp.scope)
        
        if scope_val == ScopeEnum.DETECTOR.value:
            app_config.preprocessing.detector.enabled = pp.enabled
            app_config.preprocessing.detector.clip_limit = pp.clip_limit
            app_config.preprocessing.detector.grid_size = (pp.grid_w, pp.grid_h)
            app_config.preprocessing.detector.gamma_correction = pp.gamma_correction
            app_config.preprocessing.detector.dark_threshold = pp.dark_threshold
        elif scope_val == ScopeEnum.REID.value:
            app_config.preprocessing.reid.enabled = pp.enabled
            app_config.preprocessing.reid.clip_limit = pp.clip_limit
            app_config.preprocessing.reid.grid_size = (pp.grid_w, pp.grid_h)

    print("[DB] Runtime configuration successfully linked.")
    return app_config


def ensure_areas_initialized(db: Session, app_config: AppConfig):
    """
    Seed the ``parking_areas`` table from config.yaml when it is empty.

    Mirrors :func:`ensure_config_initialized`. Areas are zoning metadata
    (B1/B2 only); when config.yaml defines no areas this is a no-op and the
    deployment runs un-zoned.
    """
    if db.query(ParkingArea).first():
        return  # already populated — DB is authoritative
    if not app_config.areas:
        return  # un-zoned deployment, nothing to seed

    print(f"[DB] Seeding {len(app_config.areas)} parking area(s) from YAML...")
    for area in app_config.areas:
        db.add(
            ParkingArea(
                area_id=area.area_id,
                name=area.name,
                floor=area.floor,
                capacity=area.capacity,
                adjacency_json=dict(area.adjacency),
                is_active=True,
            )
        )
    db.commit()
    print("[DB] Parking areas seeded successfully.")


def sync_areas_from_db(db: Session, app_config: AppConfig):
    """
    Load parking areas from the database into ``app_config.areas``.

    The DB is the runtime source of truth (mirrors
    :func:`sync_app_config_from_db`). When the table is empty the existing
    YAML-loaded areas are kept untouched.
    """
    rows = db.query(ParkingArea).filter(ParkingArea.is_active == True).all()  # noqa: E712
    if not rows:
        return app_config

    app_config.areas = [
        AreaEntry(
            area_id=row.area_id,
            name=row.name or "",
            floor=row.floor or "",
            capacity=int(row.capacity or 0),
            adjacency={
                str(k): float(v) for k, v in (row.adjacency_json or {}).items()
            },
        )
        for row in rows
    ]
    print(f"[DB] Linked {len(app_config.areas)} parking area(s) from database.")
    return app_config


def _norm_camera_id(value: str) -> str:
    """Normalize a camera id for cross-source matching (DB ``Cam_03`` vs YAML
    ``CAM-03``). Mirrors the normalization in tools/add_camera_area_column.py."""
    return (value or "").upper().replace("-", "").replace("_", "")


def load_cameras_from_db(db: Session, app_config: AppConfig):
    """
    DB-first camera roster. Replace ``app_config.cameras`` with the enabled rows
    from the API-Gateway-owned ``cameras`` table.

    Each camera's password is the decrypted ``password_encrypted`` when the
    Fernet key (``cameras_encryption_key``) is set and the token is valid;
    otherwise it falls back to the YAML password for that camera id. When the
    table is missing or has no enabled rows, the YAML-loaded cameras are kept
    untouched (mirrors :func:`sync_areas_from_db`).

    The ``cameras`` table has no ORM model (owned by the API Gateway), so it is
    queried with raw SQL guarded by ``inspect()`` — the style of
    tools/add_camera_area_column.py.
    """
    engine = db.get_bind()
    insp = inspect(engine)
    if "cameras" not in insp.get_table_names():
        print("[DB] `cameras` table not found — using YAML cameras.")
        return app_config

    # The `area` column is optional (added by add_camera_area_column.py).
    has_area = "area" in {c["name"] for c in insp.get_columns("cameras")}
    area_col = "area" if has_area else "NULL AS area"
    rows = db.execute(
        text(
            f"SELECT camera_id, name, floor, {area_col}, ip_address, rtsp_port, "
            f"username, password_encrypted, enabled FROM cameras WHERE enabled = 1"
        )
    ).fetchall()
    if not rows:
        print("[DB] No enabled cameras in DB — using YAML cameras.")
        return app_config

    decrypt = make_decrypt(app_config.cameras_encryption_key)
    # YAML fallbacks keyed by normalized camera id: password (while the Fernet
    # key is blank) and area (until cameras.area is populated, e.g. via
    # tools/add_camera_area_column.py) so DB-first doesn't drop zoning.
    yaml_pw = {_norm_camera_id(c.id): c.password for c in app_config.cameras}
    yaml_area = {_norm_camera_id(c.id): c.area for c in app_config.cameras}

    cameras = []
    decrypted_n = 0
    for r in rows:
        pw = decrypt(r.password_encrypted)
        if pw:
            decrypted_n += 1
        else:
            pw = yaml_pw.get(_norm_camera_id(r.camera_id), "")
        area = (r.area or "") or yaml_area.get(_norm_camera_id(r.camera_id), "")
        cameras.append(
            CameraEntry(
                id=r.camera_id,
                name=r.name or "",
                floor=r.floor or "",
                area=area,
                ip=r.ip_address or "",
                user=r.username or "",
                password=pw,
                rtsp_port=int(r.rtsp_port or 554),
                # Runtime slots come from the parking_slots table, keyed by
                # camera_id; slots_file is only used for one-off bootstrapping.
                slots_file="",
            )
        )

    app_config.cameras = cameras
    print(
        f"[DB] Loaded {len(cameras)} enabled camera(s) from database "
        f"({decrypted_n} password(s) decrypted, rest from YAML)."
    )
    return app_config
