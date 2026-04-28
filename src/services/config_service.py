from sqlalchemy.orm import Session
from src.model.config_run import Config, PreprocessingConfig
from src.schemas import ScopeEnum
from src.config import AppConfig

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
