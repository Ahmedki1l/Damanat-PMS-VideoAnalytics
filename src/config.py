"""
config.py — YAML configuration loader for multi-camera setup.

Loads config.yaml and provides typed access to all settings.
Supports both single-camera (legacy) and multi-camera configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class AreaEntry:
    """
    A parking *area* — a camera group within a floor (zoning).

    Zoning applies to B1/B2 only. ``adjacency`` maps a neighbouring area_id to
    the expected transit time (seconds) for a car to travel between the two
    areas; it gates the cross-area handoff matcher. ``capacity`` is the physical
    car/plate limit for the area (used as a soft cap on the bounded gallery).
    """
    area_id: str = ""
    name: str = ""
    floor: str = ""
    capacity: int = 0
    adjacency: Dict[str, float] = field(default_factory=dict)


@dataclass
class CameraEntry:
    """Configuration for a single camera."""
    id: str = ""
    name: str = ""
    floor: str = ""
    # Parking area this camera covers (zoning). Empty = un-zoned (e.g. Ground
    # floor gate cameras), which preserves the legacy all-sessions behaviour.
    area: str = ""
    ip: str = ""
    user: str = ""
    password: str = ""
    # RTSP port (from the DB cameras table; 554 keeps the legacy default).
    rtsp_port: int = 554
    slots_file: str = ""


@dataclass
class ProcessingConfig:
    """Multi-camera processing settings."""
    mode: str = "round_robin"
    target_fps_per_camera: int = 1
    stream_channel: int = 102  # 101=main(4K), 102=sub(720p)
    # Resolution that slot polygons (slots/*.json) were drawn at.
    # The runtime will scale polygon coords from this to the actual stream resolution.
    slot_ref_width: int = 1280
    slot_ref_height: int = 720
    # When True, each camera gets its own TrackedDetector so ByteTrack state
    # is not corrupted by round-robin (Ultralytics' persist=True is per-model).
    # False reverts to the legacy single shared tracker (lower RAM, worse IDs).
    per_camera_tracker: bool = True


@dataclass
class DetectorConfig:
    """YOLO detector settings."""
    model_path: str = "models/yolo11n.pt"
    confidence: float = 0.35
    classes: List[int] = field(default_factory=lambda: [2])  # 2 = car
    imgsz: int = 480
    device: str = "auto"  # "auto", "cpu", or "cuda"


@dataclass
class TrackerConfig:
    """Tracker settings."""
    type: str = "bytetrack"


@dataclass
class StateMachineConfig:
    """State machine debounce thresholds."""
    confirm_enter_frames: int = 5
    confirm_leave_frames: int = 8


@dataclass
class AssignerConfig:
    """Slot assigner settings."""
    overlap_threshold: float = 0.3


@dataclass
class DetectorPreprocessingConfig:
    """Luminance normalization settings for the detector path."""
    enabled: bool = True
    clip_limit: float = 2.0
    grid_size: tuple = (8, 8)
    gamma_correction: bool = True
    dark_threshold: int = 65


@dataclass
class ReIDPreprocessingConfig:
    """Luminance normalization settings for the ReID path."""
    enabled: bool = True
    clip_limit: float = 2.0
    grid_size: tuple = (8, 8)


@dataclass
class PreprocessingConfig:
    """Preprocessing settings for both detector and ReID."""
    detector: DetectorPreprocessingConfig = field(default_factory=DetectorPreprocessingConfig)
    reid: ReIDPreprocessingConfig = field(default_factory=ReIDPreprocessingConfig)


@dataclass
class OutputConfig:
    """Output/logging settings."""
    log_file: str = ""
    show_video: bool = False
    show_camera: str = ""  # Which camera to visualize (e.g., "CAM-04")
    snapshot_base_dir: str = "vehicle_images"
    # Externally-reachable origin used to build full snapshot URLs (e.g.
    # "http://localhost:8000"). Empty → emits site-relative URLs (legacy behaviour).
    public_base_url: str = "http://localhost:8000"
    # URL path prefix used for serving and referencing snapshot images.
    # Must match the route registered in src/api.py::create_app, which is
    # the static "/pms-video-analytics/snapshots" — overriding this in
    # yaml/env will make registry-emitted URLs miss the served route.
    snapshot_url_prefix: str = "/pms-video-analytics/snapshots"
    # Optional gateway routing prefix prepended to all generated URLs.
    # Only needed when the service sits behind a reverse-proxy / API gateway
    # (e.g. "/pms-ai" → URLs become "/pms-ai/snapshots/{file}").
    # Leave empty when accessing the service directly.
    gateway_path_prefix: str = ""

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = ""


@dataclass
class MatchingConfig:
    """
    Matching / ReID decision configuration.

    Centralises every numeric threshold and feature flag used by the matching
    cascade (see src/matching/match_decision.py). Defaults intentionally
    mirror the previously hard-coded values at
    src/vehicle_registry/vehicle_registry_identity.py so the
    matching: section is optional and behaviour is unchanged when it is
    absent from config.yaml.
    """

    # --- Cosine similarity thresholds for the B1 / global / reattach forks ---
    # B1_Entrance confirmation thresholds (lower bar for ANPR-image candidates;
    # zone-crop candidates need a higher bar).
    b1_anpr: float = 0.47
    b1_zone: float = 0.55
    # When the same plate has been seen on another camera in the gallery, drop
    # the threshold to recover cross-camera handoffs.
    b1_cross_camera: float = 0.43

    # match_global_session: default threshold callers pass in is preserved as
    # global_default. global_with_plate kicks in when the candidate session
    # already has a plate; global_cross_camera further drops the bar when the
    # session was last seen on a different camera.
    global_default: float = 0.55
    global_with_plate: float = 0.46
    global_cross_camera: float = 0.43

    # reattach_track_to_confirmed_session: default 0.52, dropped to 0.43 on
    # cross-camera reattach.
    reattach_default: float = 0.52
    reattach_cross_camera: float = 0.43

    # Legacy multi-feature fallback path (image_matcher.VehicleImageMatcher).
    legacy_color_fallback: float = 0.35

    # Dominant-color predicate floor (LAB k-means) — candidates below this
    # are hard-rejected before any ReID cosine is computed.
    color_dominant_filter: float = 0.45

    # OCR marginal band — Phase 1 WS-D wires this in. ReID scores within
    # [ocr_marginal_low, ocr_marginal_high] trigger a plate-OCR cross-check.
    ocr_marginal_low: float = 0.40
    ocr_marginal_high: float = 0.55

    # If ReID cosine clears this bar on its own, ensemble agreement is
    # optional (a "solo confirm"). Used by Phase 2 ensemble wiring.
    reid_solo_confirm: float = 0.70

    # --- HSV tolerances for color_compatible() ---
    # Tightened from 25 -> 12 in Phase 2 / T2.1: the learned 11-class color
    # classifier is now the primary path. The legacy HSV gate is kept as a
    # belt-and-braces fallback for the noop path (when no IR is loaded), so
    # the tolerance can be much tighter without dropping legitimate matches —
    # the rare colour-classifier-confused case is now covered by the K-of-N
    # ensemble rule rather than a wide HSV envelope.
    hsv_h_tol: float = 12.0
    hsv_s_tol: float = 80.0
    hsv_v_tol: float = 80.0

    # --- Feature flags (previously env vars) ---
    use_color_filter: bool = False
    use_lab_clahe: bool = False
    use_multishot: bool = False

    # --- Ensemble / voting (Phase 2 wiring; defaults make them inert) ---
    ensemble_min_modalities_agree: int = 2
    reid_solo_confirm_threshold: float = 0.70
    voting_enabled: bool = False
    voting_window_frames: int = 5
    voting_min_agree: int = 3

    # --- Plugin model paths (Phase 1) ---
    # ``color_classifier_model``: Path to the OpenVINO IR (``model.xml``) for
    # the WS-B color classifier. Default points at the artifact directory
    # written by ``tools/train_color_classifier.py``. When the file is
    # missing, the plugin raises a clear RuntimeError on first predict().
    color_classifier_model: str = "models/color_classifier_openvino/model.xml"
    type_classifier_model: str = ""
    plate_ocr_model: str = ""

    # --- Fast ReID backend (Phase 1 / WS-A) ---
    # When ``use_openvino_reid`` is on AND an OpenVINO IR for OSNet exists at
    # ``models/osnet_openvino_int8/model.xml`` the matcher uses the OpenVINO
    # runtime path (target ≤40 ms/image on CPU). When the file is missing or
    # the flag is off the matcher falls back to the legacy torchreid path
    # (~1 s/image on CPU). ``reid_input_size`` is ``(height, width)`` and
    # must agree with the size baked into the exported IR.
    use_openvino_reid: bool = True
    reid_input_size: tuple = (192, 96)
    reid_openvino_model_dir: str = "models/osnet_openvino_int8"

    # --- Phase 3 / T3.2 — FAISS-CPU gallery index ---
    # ``match_global_session`` defaults to the legacy O(n) linear scan
    # over ``self._sessions``. When ``use_faiss_index`` is True and the
    # ``faiss-cpu`` package is importable, the registry instead routes the
    # query through ``src.matching.GalleryIndex`` for an O(log n) IVF
    # search. Default is False until the production rollout completes its
    # shadow-mode benchmarking. ``faiss_index_dimension`` MUST match the
    # ReID feature length (512 for OSNet-AIN).
    use_faiss_index: bool = False
    faiss_index_dimension: int = 512
    faiss_index_nlist: int = 8


@dataclass
class AlertsConfig:
    """Alerting feature toggles."""
    enable_restricted_zone_alerts: bool = False


@dataclass
class AppConfig:
    """Root configuration container."""
    cameras: List[CameraEntry] = field(default_factory=list)
    # Parking areas (zoning). Empty on un-zoned deployments — every helper and
    # the bounded matcher degrade to the legacy all-sessions behaviour then.
    areas: List[AreaEntry] = field(default_factory=list)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    assigner: AssignerConfig = field(default_factory=AssignerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)

    # Fernet key (urlsafe-base64, 32 bytes) shared with the API Gateway to
    # decrypt cameras.password_encrypted. Blank = fall back to YAML passwords.
    cameras_encryption_key: str = ""

    # Legacy single-camera support
    video_source: str = ""
    video_target_fps: int = 2
    slots_file: str = "parking_slots.json"

    @property
    def is_multi_camera(self) -> bool:
        """True if config has multiple cameras defined."""
        return len(self.cameras) > 0

    # --- Zoning helpers -------------------------------------------------
    def area_for_camera(self, camera_id: str) -> str:
        """Return the area_id a camera belongs to, or "" if un-zoned."""
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam.area
        return ""

    def cameras_in_area(self, area_id: str) -> List[str]:
        """Return the camera ids assigned to an area."""
        if not area_id:
            return []
        return [cam.id for cam in self.cameras if cam.area == area_id]

    def area_by_id(self, area_id: str) -> Optional[AreaEntry]:
        """Return the AreaEntry for an area_id, or None if undefined."""
        for area in self.areas:
            if area.area_id == area_id:
                return area
        return None

    def adjacency_for(self, area_id: str) -> Dict[str, float]:
        """Return {neighbor_area_id: transit_seconds} for an area (empty if none)."""
        area = self.area_by_id(area_id)
        return dict(area.adjacency) if area else {}


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file.

    Supports both multi-camera format and legacy single-camera format.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Populated AppConfig instance.
    """
    config = AppConfig()


    if not os.path.exists(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Using defaults.")
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    if "database" in raw:
        config.database.url = raw["database"].get("DATABASE_URL", "")

    # Fernet key for decrypting cameras.password_encrypted (env overrides YAML).
    config.cameras_encryption_key = (
        raw.get("CAMERAS_ENCRYPTION_KEY", "")
        or os.environ.get("CAMERAS_ENCRYPTION_KEY", "")
    )

    # --- Cameras (multi-camera mode) ---
    if "cameras" in raw:
        for cam_raw in raw["cameras"]:
            cam = CameraEntry(
                id=cam_raw.get("id", ""),
                name=cam_raw.get("name", ""),
                floor=cam_raw.get("floor", ""),
                area=cam_raw.get("area", ""),
                ip=cam_raw.get("ip", ""),
                user=cam_raw.get("user", ""),
                password=cam_raw.get("password", ""),
                slots_file=cam_raw.get("slots_file", ""),
            )
            config.cameras.append(cam)

    # --- Areas (zoning; optional — absent on un-zoned deployments) ---
    if "areas" in raw:
        for area_raw in raw["areas"] or []:
            adj_raw = area_raw.get("adjacency", {}) or {}
            # adjacency: {neighbor_area_id: transit_seconds}
            adjacency = {str(k): float(v) for k, v in adj_raw.items()}
            area = AreaEntry(
                area_id=area_raw.get("area_id", area_raw.get("id", "")),
                name=area_raw.get("name", ""),
                floor=area_raw.get("floor", ""),
                capacity=int(area_raw.get("capacity", 0) or 0),
                adjacency=adjacency,
            )
            config.areas.append(area)

    # --- Processing ---
    if "processing" in raw:
        p = raw["processing"]
        config.processing.mode = p.get("mode", config.processing.mode)
        config.processing.target_fps_per_camera = p.get(
            "target_fps_per_camera", config.processing.target_fps_per_camera
        )
        config.processing.stream_channel = p.get(
            "stream_channel", config.processing.stream_channel
        )
        config.processing.slot_ref_width = p.get(
            "slot_ref_width", config.processing.slot_ref_width
        )
        config.processing.slot_ref_height = p.get(
            "slot_ref_height", config.processing.slot_ref_height
        )
        config.processing.per_camera_tracker = bool(p.get(
            "per_camera_tracker", config.processing.per_camera_tracker
        ))

    # --- Legacy single-camera support ---
    if "video" in raw:
        v = raw["video"]
        config.video_source = v.get("source", "")
        config.video_target_fps = v.get("target_fps", 2)
    if "slots" in raw:
        config.slots_file = raw["slots"].get("file", "parking_slots.json")

    # --- Detector ---
    if "detector" in raw:
        d = raw["detector"]
        config.detector.model_path = d.get("model_path", config.detector.model_path)
        config.detector.confidence = d.get("confidence", config.detector.confidence)
        config.detector.classes = d.get("classes", config.detector.classes)
        config.detector.imgsz = d.get("imgsz", config.detector.imgsz)
        config.detector.device = d.get("device", config.detector.device)

    # --- Tracker ---
    if "tracker" in raw:
        config.tracker.type = raw["tracker"].get("type", config.tracker.type)

    # --- State Machine ---
    if "state_machine" in raw:
        sm = raw["state_machine"]
        config.state_machine.confirm_enter_frames = sm.get(
            "confirm_enter_frames", config.state_machine.confirm_enter_frames
        )
        config.state_machine.confirm_leave_frames = sm.get(
            "confirm_leave_frames", config.state_machine.confirm_leave_frames
        )

    # --- Assigner ---
    if "assigner" in raw:
        config.assigner.overlap_threshold = raw["assigner"].get(
            "overlap_threshold", config.assigner.overlap_threshold
        )

    # --- Preprocessing ---
    if "preprocessing" in raw:
        pp = raw["preprocessing"]
        if "detector" in pp:
            d_pp = pp["detector"]
            config.preprocessing.detector.enabled = d_pp.get(
                "enabled", config.preprocessing.detector.enabled
            )
            config.preprocessing.detector.clip_limit = d_pp.get(
                "clip_limit", config.preprocessing.detector.clip_limit
            )
            gs = d_pp.get("grid_size", None)
            if gs is not None:
                config.preprocessing.detector.grid_size = tuple(gs)
            config.preprocessing.detector.gamma_correction = d_pp.get(
                "gamma_correction", config.preprocessing.detector.gamma_correction
            )
            config.preprocessing.detector.dark_threshold = d_pp.get(
                "dark_threshold", config.preprocessing.detector.dark_threshold
            )
        if "reid" in pp:
            r_pp = pp["reid"]
            config.preprocessing.reid.enabled = r_pp.get(
                "enabled", config.preprocessing.reid.enabled
            )
            config.preprocessing.reid.clip_limit = r_pp.get(
                "clip_limit", config.preprocessing.reid.clip_limit
            )
            gs = r_pp.get("grid_size", None)
            if gs is not None:
                config.preprocessing.reid.grid_size = tuple(gs)

    # --- Matching (ReID decision thresholds + plugin paths) ---
    # Read env-var legacy flags so existing deployments behave the same when
    # the new matching: block is absent from config.yaml. config.yaml wins
    # over env vars when both are set — yaml is the documented source of truth.
    config.matching.use_color_filter = (
        os.environ.get("REID_USE_COLOR_FILTER", "false").lower() == "true"
    )
    config.matching.use_lab_clahe = (
        os.environ.get("REID_USE_LAB_CLAHE", "false").lower() == "true"
    )
    config.matching.use_multishot = (
        os.environ.get("REID_USE_MULTISHOT", "false").lower() == "true"
    )

    # Alert toggles — env-var first, YAML overrides below if present.
    config.alerts.enable_restricted_zone_alerts = (
        os.environ.get("ENABLE_RESTRICTED_ZONE_ALERTS", "false").lower() == "true"
    )
    if "alerts" in raw:
        a = raw["alerts"] or {}
        config.alerts.enable_restricted_zone_alerts = a.get(
            "enable_restricted_zone_alerts",
            config.alerts.enable_restricted_zone_alerts,
        )

    if "matching" in raw:
        m = raw["matching"] or {}
        cm = config.matching

        # Thresholds
        cm.b1_anpr = m.get("b1_anpr", cm.b1_anpr)
        cm.b1_zone = m.get("b1_zone", cm.b1_zone)
        cm.b1_cross_camera = m.get("b1_cross_camera", cm.b1_cross_camera)
        cm.global_default = m.get("global_default", cm.global_default)
        cm.global_with_plate = m.get("global_with_plate", cm.global_with_plate)
        cm.global_cross_camera = m.get("global_cross_camera", cm.global_cross_camera)
        cm.reattach_default = m.get("reattach_default", cm.reattach_default)
        cm.reattach_cross_camera = m.get(
            "reattach_cross_camera", cm.reattach_cross_camera
        )
        cm.legacy_color_fallback = m.get(
            "legacy_color_fallback", cm.legacy_color_fallback
        )
        cm.color_dominant_filter = m.get(
            "color_dominant_filter", cm.color_dominant_filter
        )
        cm.ocr_marginal_low = m.get("ocr_marginal_low", cm.ocr_marginal_low)
        cm.ocr_marginal_high = m.get("ocr_marginal_high", cm.ocr_marginal_high)
        cm.reid_solo_confirm = m.get("reid_solo_confirm", cm.reid_solo_confirm)

        # HSV tolerances
        cm.hsv_h_tol = m.get("hsv_h_tol", cm.hsv_h_tol)
        cm.hsv_s_tol = m.get("hsv_s_tol", cm.hsv_s_tol)
        cm.hsv_v_tol = m.get("hsv_v_tol", cm.hsv_v_tol)

        # Feature flags — yaml overrides env-var defaults set above
        cm.use_color_filter = m.get("use_color_filter", cm.use_color_filter)
        cm.use_lab_clahe = m.get("use_lab_clahe", cm.use_lab_clahe)
        cm.use_multishot = m.get("use_multishot", cm.use_multishot)

        # Ensemble / voting (Phase 2)
        cm.ensemble_min_modalities_agree = m.get(
            "ensemble_min_modalities_agree", cm.ensemble_min_modalities_agree
        )
        cm.reid_solo_confirm_threshold = m.get(
            "reid_solo_confirm_threshold", cm.reid_solo_confirm_threshold
        )
        cm.voting_enabled = m.get("voting_enabled", cm.voting_enabled)
        cm.voting_window_frames = m.get(
            "voting_window_frames", cm.voting_window_frames
        )
        cm.voting_min_agree = m.get("voting_min_agree", cm.voting_min_agree)

        # Plugin model paths (Phase 1)
        cm.color_classifier_model = m.get(
            "color_classifier_model", cm.color_classifier_model
        )
        cm.type_classifier_model = m.get(
            "type_classifier_model", cm.type_classifier_model
        )
        cm.plate_ocr_model = m.get("plate_ocr_model", cm.plate_ocr_model)

        # Fast ReID backend (WS-A)
        cm.use_openvino_reid = m.get("use_openvino_reid", cm.use_openvino_reid)
        ris = m.get("reid_input_size", None)
        if ris is not None:
            # Accept "HxW" string or [H, W] list/tuple. Stored as (H, W).
            if isinstance(ris, str) and "x" in ris.lower():
                h_str, w_str = ris.lower().split("x", 1)
                cm.reid_input_size = (int(h_str), int(w_str))
            elif isinstance(ris, (list, tuple)) and len(ris) == 2:
                cm.reid_input_size = (int(ris[0]), int(ris[1]))
        cm.reid_openvino_model_dir = m.get(
            "reid_openvino_model_dir", cm.reid_openvino_model_dir
        )

        # FAISS-CPU gallery index (Phase 3 / T3.2)
        cm.use_faiss_index = m.get("use_faiss_index", cm.use_faiss_index)
        cm.faiss_index_dimension = m.get(
            "faiss_index_dimension", cm.faiss_index_dimension
        )
        cm.faiss_index_nlist = m.get("faiss_index_nlist", cm.faiss_index_nlist)

    # --- Output ---
    if "output" in raw:
        o = raw["output"]
        config.output.log_file = o.get("log_file", config.output.log_file)
        config.output.show_video = o.get("show_video", config.output.show_video)
        config.output.show_camera = o.get("show_camera", config.output.show_camera)
        config.output.snapshot_base_dir = o.get(
            "snapshot_base_dir",
            os.environ.get("SNAPSHOT_PATH", config.output.snapshot_base_dir)
        )
        config.output.public_base_url = o.get(
            "public_base_url",
            os.environ.get("PUBLIC_BASE_URL", config.output.public_base_url),
        )
        config.output.snapshot_url_prefix = o.get(
            "snapshot_url_prefix",
            config.output.snapshot_url_prefix,
        )
        config.output.gateway_path_prefix = o.get(
            "gateway_path_prefix",
            os.environ.get("GATEWAY_PATH_PREFIX", config.output.gateway_path_prefix),
        )

    return config
