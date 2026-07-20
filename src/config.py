"""
config.py — YAML configuration loader for multi-camera setup.

Loads config.yaml and provides typed access to all settings.
Supports both single-camera (legacy) and multi-camera configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


def norm_camera_id(value: str) -> str:
    """Normalize a camera id for cross-source matching (DB ``Cam_03`` vs YAML
    ``CAM-03``). Mirrors the normalization in services/config_service.py."""
    return (value or "").upper().replace("-", "").replace("_", "")


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
    # Cap how often each camera materializes/publishes a drained RTSP frame.
    # The capture still consumes the compressed stream continuously so FFmpeg
    # cannot build a stale backlog. 0 publishes every source frame.
    max_grab_fps: int = 0
    # Resolution that slot polygons (slots/*.json) were drawn at.
    # The runtime will scale polygon coords from this to the actual stream resolution.
    slot_ref_width: int = 1280
    slot_ref_height: int = 720
    # When True, each camera gets its own TrackedDetector so ByteTrack state
    # is not corrupted by round-robin (Ultralytics' persist=True is per-model).
    # False reverts to the legacy single shared tracker (lower RAM, worse IDs).
    per_camera_tracker: bool = True


# Fields a detector.camera_overrides entry is allowed to set. Deliberately
# excludes device / ov_* : those are properties of the HOST, not the camera, and
# OpenVINO sizes its thread pool once per process — letting one camera reset them
# would silently retune every other camera in the same group. "preprocessing" is
# a nested dict validated against PREPROCESSING_OVERRIDE_KEYS.
DETECTOR_OVERRIDE_KEYS = frozenset(
    {"model_path", "confidence", "imgsz", "classes", "max_box_area_ratio",
     "preprocessing"}
)

# Fields a camera override's nested "preprocessing" block may set — the CLAHE
# knobs on DetectorPreprocessingConfig.
PREPROCESSING_OVERRIDE_KEYS = frozenset(
    {"enabled", "clip_limit", "grid_size", "gamma_correction", "dark_threshold",
     "detector_scale"}
)


def _parse_detector_overrides(raw: Dict) -> Dict[str, Dict]:
    """Parse+validate detector.camera_overrides. Unknown keys raise rather than
    being dropped: a typo here fails open (camera keeps the broken global model)
    and the symptom — no detections — is exactly what the override was added to
    fix, so it would be near-impossible to spot in the field."""
    parsed: Dict[str, Dict] = {}
    for cam_id, fields in (raw or {}).items():
        fields = dict(fields or {})
        unknown = set(fields) - DETECTOR_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"detector.camera_overrides['{cam_id}']: unknown key(s) "
                f"{sorted(unknown)}. Allowed: {sorted(DETECTOR_OVERRIDE_KEYS)}"
            )
        pp = fields.get("preprocessing")
        if pp is not None:
            pp_unknown = set(pp or {}) - PREPROCESSING_OVERRIDE_KEYS
            if pp_unknown:
                raise ValueError(
                    f"detector.camera_overrides['{cam_id}'].preprocessing: unknown "
                    f"key(s) {sorted(pp_unknown)}. Allowed: "
                    f"{sorted(PREPROCESSING_OVERRIDE_KEYS)}"
                )
            fields["preprocessing"] = dict(pp or {})
        if fields:
            parsed[norm_camera_id(cam_id)] = fields
    return parsed


@dataclass
class DetectorConfig:
    """YOLO detector settings."""
    model_path: str = "models/yolo11n.pt"
    confidence: float = 0.35
    classes: List[int] = field(default_factory=lambda: [2])  # 2 = car
    imgsz: int = 480
    device: str = "auto"  # "auto", "cpu", or "cuda"

    # Drop any box covering more than this fraction of the frame. 0.0 = off.
    # Large models occasionally emit a whole-frame "car"/"bus"; the slot
    # assigner's primary path is bottom-center point-in-polygon, so such a box
    # can land in a slot and pin it OCCUPIED. Real vehicles here are <10% of
    # frame even on the fisheye, so this only ever fires on the bogus box.
    max_box_area_ratio: float = 0.0

    # YAML-ONLY per-camera overrides, keyed by normalized camera id (see
    # norm_camera_id). NOT DB-owned: sync_app_config_from_db rewrites
    # model_path/confidence/imgsz from the Config table on every sync, so a
    # per-camera value stored there would be clobbered — it lives in YAML alone,
    # same as ov_num_threads above. Each entry may set any of
    # DETECTOR_OVERRIDE_KEYS; unset keys inherit the global (DB) value.
    camera_overrides: Dict[str, Dict] = field(default_factory=dict)

    # OpenVINO CPU pool. NOT DB-owned — these two are YAML-only (see
    # config_service.sync_app_config_from_db, which never touches them).
    #
    # Ultralytics hardcodes PERFORMANCE_HINT=LATENCY, and on CPU that hint caps
    # the pool at 8 threads no matter how many cores exist — which is why a
    # 15-core box shows ~8 busy cores and looks half-idle. 0 / "" keeps that
    # default. See src/ov_tuning.py for the measured cost/benefit before raising
    # it: widening is worth ~1.2x and REGRESSES past ~16 threads.
    ov_num_threads: int = 0          # 0 = leave OpenVINO's default
    ov_performance_hint: str = ""    # "" = leave LATENCY; or THROUGHPUT


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
    # Run CLAHE at the detector's input size instead of the full frame. The IR is
    # statically shaped (320x320), so a 1280x720 frame gets letterboxed down to
    # 320x180 inside Ultralytics anyway — normalising beforehand enhances ~16x
    # more pixels than the model ever sees. Resizing first makes the whole
    # BGR<->LAB round trip cheap, and the detector's input is unchanged apart
    # from CLAHE being computed on the downscaled pixels.
    # Set false to go back to full-frame CLAHE if detection recall regresses.
    detector_scale: bool = True


@dataclass
class ReIDPreprocessingConfig:
    """Luminance normalization settings for the ReID path.

    NOTE: applies to the legacy torchreid fallback ONLY. The production
    OpenVINO backend never applies CLAHE — its IRs (e.g. PS_carMatching) are
    trained and calibrated on raw squish-resized crops, and normalising
    luminance at inference would shift embeddings off the trained
    distribution (see VehicleReIDMatcher backend selection logging)."""
    enabled: bool = False
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

    # Cameras whose anonymous tracks must NOT donate their appearance into a
    # confirmed session via reattach. A static parking-slot camera permanently
    # frames the same parked car; it endlessly mints fresh anonymous tracks that
    # hunt for any confirmed session scoring above reattach_cross_camera (0.43)
    # and, on a borderline match, poison that session's gallery with a different
    # car (the CAM-24 champagne-Lexus-into-a-dark-Hyundai leak). Listed cameras
    # still detect, track locally, and can be identified by the forward
    # global-match path — they just cannot ADOPT another car's identity by
    # reattach. Same spirit as the gate-camera exclusion. Empty ⇒ legacy.
    reattach_excluded_cameras: tuple = ()

    # Abstain-on-ambiguity margin for match_global_session: when the best and
    # second-best candidates score within this distance of each other, the
    # match is ambiguous (visually similar cars) and NO session is returned —
    # a missing label is recoverable on a later frame, a wrong plate is not.
    # 0 disables the gate.
    global_match_margin: float = 0.05

    # --- Per-car persistent multi-shot gallery (folder per plate) --------- #
    # A growing set of quality-gated reference crops+vectors per plate, kept on
    # disk so a car's appearance profile survives restart and warm-starts a
    # returning car. All gated by ``gallery_persist_enabled`` — off ⇒ today's
    # in-memory-only behaviour.
    gallery_persist_enabled: bool = False
    gallery_max_refs_per_car: int = 10        # cap crops/vectors per plate folder
    gallery_retention_days: float = 5.0       # GC folders idle longer than this
    gallery_min_view_quality: float = 0.9     # full-view gate (see _bbox_view_quality)
    # D9 neighbour-clearance: a car parked shoulder-to-shoulder has a bbox that a
    # neighbour's box overlaps, so its ReID crop is contaminated with the wrong
    # car. When enforced, _bbox_view_quality is multiplied by the clearance
    # (1.0 unobstructed → 0.0 fully covered), so an occluded crop falls below
    # gallery_min_view_quality and is kept out of the gallery. Default OFF
    # (log-only): the clearance is computed and logged to collect its
    # distribution BEFORE it is allowed to gate anything.
    gallery_neighbour_clearance_enforce: bool = False
    # Slot authority: a camera may only add a gallery reference for a car that is
    # inside a slot IT hosts. Owning a car was previously the only camera check on
    # the teach path, so an aisle camera that won the ownership contest could write
    # references for a car parked in a DIFFERENT camera's slot — which is how four
    # crops of a black Ford entered grey-Hyundai DJS-7842's gallery from CAM-07
    # (whose own slot is 4% of a frame whose ROI is 74%). Set False to restore the
    # old any-owner-may-teach behaviour.
    gallery_require_slot_authority: bool = True
    # --- Slot acquisition by elimination -------------------------------------
    # Appearance alone CANNOT bridge the gate->slot viewpoint gap. A car's first view
    # from its slot camera is a small oblique crop (~136px tall) matched against its
    # front-gate photos: measured 0.583 for RDJ-9640 against its OWN references, below
    # the 0.62 bar. So the car fails to recognise itself on arrival, mints an anonymous
    # session, and parks as plate=(none) forever. Lowering the bar does NOT fix it —
    # from that viewpoint a DIFFERENT car scored 0.634, i.e. the ranking is inverted,
    # so a looser bar binds the WRONG plate.
    #
    # Geography and timing can identify the car where appearance cannot. Within the
    # AREA-SCOPED candidate pool, if exactly ONE plated car is still "in flight"
    # (entered recently, not yet linked to any slot), it is the only car this can be.
    # Appearance is then only required not to CONTRADICT — hence a floor, well clear of
    # the negative-pair distribution (p10 0.27 / p50 0.39), rather than a decision bar.
    #
    # Four independent constraints must agree before a plate is bound this way:
    # right area, still in transit, uniquely the only candidate, appearance not objecting.
    # ONLY A READ PLATE MAY REACH parking_slots.current_plate.
    # The ReID slot-binding path (try_link_to_slot) binds on APPEARANCE, and appearance
    # at the slot is INVERTED: measured 2026-07-11, a car scored 0.583 against its OWN
    # gate references while a DIFFERENT car scored 0.634. That path stamped RDJ-9640 onto
    # B1_CRO (which held DJS-7842) and tore RDJ-9640's plate off its real slot in the
    # same move. OCR reads the characters off the car instead, and binds only what
    # exactly one car inside can explain � so current_plate is CORRECT or NULL, never
    # wrong. Set False to restore the old appearance-based binding.
    # Default False so the library's legacy behaviour is unchanged; production opts IN
    # via config.yaml. (Flipping the dataclass default silently rewires every caller and
    # broke 13 tests that legitimately exercise the ReID slot-binding path.)
    slot_plate_requires_ocr: bool = False
    # THE TRANSIT HOP. Every car entering this facility passes CAM-20, which reads plates
    # reliably. A camera that CANNOT read one (CAM-21 is mounted side-on to its aisle and
    # has produced zero reads, parked or moving) adopts the identity of the single car
    # that was READ moments ago and has not parked yet � then captures the SIDE-VIEW
    # reference the gallery has never had. From that car's second visit, ReID matches
    # side-vs-side (~0.9) and this hop is never needed again.
    slot_acquire_by_ocr_transit: bool = True
    ocr_transit_window_seconds: float = 120.0
    slot_acquire_by_elimination: bool = True
    slot_acquire_min_similarity: float = 0.50   # appearance must not CONTRADICT
    slot_acquire_inflight_seconds: float = 300.0  # gate -> slot transit window
    gallery_min_sharpness: float = 40.0       # reject blurry crops (sharpness_score)
    gallery_dedup_cosine: float = 0.97        # skip near-duplicate refs
    gallery_accumulate_interval_s: float = 3.0  # throttle per (plate, camera)
    # Feedback-loop guard: a new floor-camera reference must resemble the car's
    # GROUND-TRUTH references (ANPR/CAM-03/CAM-23 primary + refs), scoring at
    # least this cosine against the best of them, before it is LEARNED as a
    # gallery reference. Applies to BOTH the disk-accumulate path and the reattach
    # append path (association can happen at reattach_cross_camera 0.43, but
    # LEARNING an appearance requires clearing this higher bar). Legitimate
    # oblique cross-view crops still pass; a wrong-car crop that slipped past
    # matching (0.43-0.60) does not. 0.30 proved a formality — a night crop of a
    # different car clears it — so the floor sits at 0.45. Fail-open when the
    # session has no ground-truth reference yet.
    gallery_accumulate_min_gt_similarity: float = 0.45
    # Minimum reference-crop resolution (pixels, width*height). A distant car
    # occupies too few pixels to make a useful ReID reference — upscaling it to
    # the model input is mostly interpolation — and lets a far camera flood a
    # plate's gallery with tiny crops. Reject crops below this area so only
    # close, well-resolved views are stored. (A 110x110 car ~= 12k px.)
    gallery_min_crop_area: float = 12000.0

    # CAM-03 confirmation-timeline size. The confirmation burst captures the car
    # front-to-rear as it crosses the B-entry zone; the timeline gallery emits
    # entry (front) + deep (primary) + exit (rear) plus up to this many EXTRA
    # rear-side frames sampled from the crossing tail, so the car's BACK is
    # represented by more than a single far exit frame. Capped downstream by
    # gallery_max_refs_per_car and the cosine-dedup, so redundant frames are
    # dropped. Set 0 to keep the legacy fixed entry/deep/exit triple.
    confirmation_extra_rear_views: int = 3

    # Source-camera trust weighting for ReID gallery references. The parking has
    # three GROUND-TRUTH cameras whose crops are canonical viewpoints — ANPR
    # (front), CAM-23 (top), CAM-03 (upper-front + back). A reference captured by
    # one of these scores at full weight; a reference from ANY other camera (an
    # oblique in-garage side view) has its similarity multiplied by
    # ``secondary_camera_weight`` (<1), so a secondary crop can only win a match
    # when it is substantially stronger than every ground-truth view. The session
    # PRIMARY (feature_vector, always set from a confirmation/seed path) is always
    # full weight. A reference with no recorded source camera is treated as
    # secondary (conservative default). Set weight = 1.0 for uniform/legacy.
    ground_truth_cameras: tuple = ("ANPR", "ANPR-Entry", "ANPR-Exit", "CAM-23", "CAM-03")
    secondary_camera_weight: float = 0.6

    # ...with ONE exception, on the slot path. When identifying a car parked in a slot,
    # a reference captured by THAT SAME slot camera is not an oblique guess — it is the
    # car photographed in the very pose we are looking at, taught by an earlier
    # OCR-confirmed park (save_parked_reference). At 0.6 a car's own parked pose (cosine
    # ~0.9 -> 0.54) loses to a DIFFERENT car's full-weight gate photo, so the car cannot
    # be recognised even on a return visit.
    #
    # But it must NOT go to 1.0 either. The boost is symmetric — it lifts every
    # candidate's same-camera refs, including the regulars who park at this camera daily
    # — and same-view similarity between DIFFERENT cars (~0.66) exceeds cross-view
    # similarity for the SAME car (~0.55). At full weight those regulars outrank a car
    # that has never parked here. Suppressing that is what the 0.6 discount was FOR.
    #
    # Swept on the live 34-car gallery, 2026-07-13 (263 real parked-pose queries), with
    # the query crop withheld. WARM = the car has parked at this camera before;
    # COLD = it never has:
    #     weight   WARM rank-1   COLD rank-1
    #     0.60         98.5%        87.8%    <- old behaviour (no special case)
    #     0.70        100.0%        87.8%
    #     0.80        100.0%        87.8%    <- chosen: full warm gain, zero cold cost
    #     0.90        100.0%        87.5%
    #     1.00        100.0%        87.5%    <- naive "full weight"; costs cold for nothing
    # 0.80 sits mid-plateau, so neither bound is near a cliff. Re-derive on any ReID
    # model swap — PS_matcher V2's cosine scale is not V1's.
    # Slot path only; the gate/global path keeps the plain 0.6, where it is load-bearing.
    slot_camera_ref_weight: float = 0.80

    # ---- ReID-solo fallback -------------------------------------------------------
    # Some slots can NEVER be read. CAM-21 frames B1_CRO in pure side profile: 455
    # attempts, zero reads. On those, insisting on an OCR witness means the slot stays
    # NULL forever, so appearance has to stand alone — but only when it is genuinely sure.
    #
    # `reid_solo_confirm` (0.90) is NOT reused here: on PS_matcher V2 it accepts 0 of 311
    # real queries. It was calibrated for the gate/global comparison on a model whose
    # cosine scale was much wider. Borrowing it would silently disable this path.
    #
    # THE TWO GATES GUARD DIFFERENT FAILURES. Both are required.
    #
    # MARGIN — guards against confusing two cars we KNOW. Derived from 311 real
    # parked-pose queries, 2026-07-13:
    #
    #                wrong top-1   worst wrong score   worst wrong margin
    #     WARM            2%             0.714               0.094
    #     COLD           26%             0.762               0.099
    #
    # A wrong car never wins by more than 0.099, so margin >= 0.15 clears it with >= 0.05
    # headroom in BOTH regimes — the same headroom rule that derived lock_confidence.
    # Zero false accepts over the whole set.
    #
    # SCORE — guards against a car we have NEVER SEEN. This is the one the offline eval
    # cannot measure, because that eval is CLOSED-SET: the true car is always somewhere in
    # the gallery, so "the right answer is absent" never happens and the score floor looks
    # redundant (margin>=0.15 alone: 42 accepted, 0 wrong; adding score>=0.70 changes
    # nothing). Production is OPEN-SET, and it shows the real failure immediately:
    #
    #     B14  best=RDJ-9640  score 0.600  margin 0.356
    #     B19  best=ERS-7949  score 0.669  margin 0.411
    #     B22  best=ZRS-6511  score 0.615  margin 0.388
    #
    # LOW score with a WIDE margin is the signature of an unknown vehicle: nothing matches
    # it well, but one gallery car is the nearest of a bad bunch, so it wins by a mile.
    # Margin alone would confidently stamp a stranger with RDJ-9640's plate. The floor is
    # what refuses. (Those cars are genuinely absent — 17 registered vehicles have no open
    # session; B28's is HRS-4772, B25's is AVD-4918, neither ever seen at the gate.)
    #
    # WHERE THE FLOOR SITS. Cold cars score lower than warm ones — there is no
    # parked-pose reference yet, only a gate photo, and that is the cross-view case.
    # DAY-1 COLD, top-1 score:
    #     CORRECT  p5=0.614  p10=0.637  p25=0.682  p50=0.737
    #     WRONG              p50=0.646             max=0.762
    # The two overlap heavily, so the score is a weak correct/wrong discriminator — the
    # margin does that job. What the score DOES separate is the open-set case: an unknown
    # car matches nothing well (B25 scored 0.269, B28 0.327), far below any real match.
    #
    # The floor is set by the GAP between a known car and an unknown one, and production
    # ground truth is what pins it. Two solo answers have been verified against the plate
    # visible in the frame, and ReID was right BOTH times:
    #
    #     B19  ERS-7949  score 0.669-0.721   CORRECT (plate reads 7949 ERS)
    #     B14  RDJ-9640  score 0.586-0.600   CORRECT (operator-confirmed)
    #
    # ...while the cars that are genuinely NOT enrolled score far lower, because nothing
    # in the gallery resembles them:
    #
    #     B25  best 0.218-0.281      B28  best 0.304-0.339
    #
    # So known-cold lands at 0.59+, unknown at <=0.34 — a wide, clean gap. 0.50 sits in
    # the middle of it, with 0.16 of headroom over the worst unknown and 0.086 under the
    # weakest verified-correct answer.
    #
    # Earlier floors of 0.70 (from WARM statistics) and 0.65 were BOTH too high and were
    # silently discarding correct answers: cold cars have no parked-pose reference yet,
    # only a gate photo, so their scores sit lower by construction (correct-cold p10 =
    # 0.637, p5 = 0.614). Do not re-raise this from warm numbers.
    #
    # The margin gate remains the guard against confusing two KNOWN cars; the floor is
    # purely the open-set guard. Both are needed — see the note above.
    slot_reid_solo_enabled: bool = True
    slot_reid_solo_after_attempts: int = 8      # give OCR most of its budget first
    slot_reid_solo_min_score: float = 0.50
    slot_reid_solo_min_margin: float = 0.15
    # Reserved slots stay stricter: a wrong plate here raises a false intrusion alert
    # against an executive, so precision beats coverage. The MARGIN is what is tightened
    # (0.20 vs 0.15) — that is the guard that actually discriminates. B1_CRO's best guess
    # scores a respectable 0.677 but wins by only 0.047, and is correctly refused.
    slot_reid_solo_min_score_reserved: float = 0.60
    slot_reid_solo_min_margin_reserved: float = 0.20

    # Keep asking appearance, every N seconds, while a slot is occupied and still
    # nameless. The 12-attempt budget covers the parking manoeuvre and is then spent
    # forever, which on a never-readable slot gave ReID five shots ~4s apart on ONE pose
    # in ONE light. The gallery is not static — a car gains references as it is seen
    # elsewhere — so the same query can clear the margin later. Costs one embedding
    # (~15ms), no OCR. 0 disables the retry.
    #
    # DELIBERATELY SLOW. Re-asking a noisy scorer will eventually cross any threshold by
    # luck, and these slots have no OCR witness that could ever catch a wrong answer, so
    # the margin gate is all that protects the customer. A minute samples genuinely
    # changed conditions; seconds would just resample the same frame until it got lucky.
    slot_reid_retry_interval_s: float = 60.0

    # Slots whose plate is NEVER in frame — the camera films them side-on or at
    # point-blank range (verified 2026-07-15: CAM-13 fills 87% of frame with B22's flank;
    # CAM-21 has logged 455 attempts / zero reads on B1_CRO). On these, OCR is not a
    # slow path, it is a DEAD path: it burns 12 x ~200-670ms of the frame loop per park
    # to read the wall, the floor, or the camera's own OSD watermark, and then hands the
    # slot to appearance anyway. Listing a slot here skips OCR entirely and lets ReID
    # answer from the first attempt, on the SAME score/margin gates as everywhere else.
    # This is a statement of physical fact about a camera's view, so it is operator-set
    # (config.yaml `matching.slot_no_plate_view`), never inferred.
    slot_no_plate_view: List[str] = field(default_factory=list)

    # Minimum seconds between ANY two slot-identify OCR reads in one worker
    # PROCESS (the per-slot 0.7s interval alone cannot pace the loop: with ~10
    # armed slots some slot is always eligible, so every frame pays a read).
    # A PaddleOCR read is 2-8 SECONDS on the production Xeon and runs INLINE in
    # the serial camera loop — unpaced, a post-restart arming storm took b2-a to
    # 19s/frame (measured 2026-07-16: ocr=8.8s of every frame, 50 frames in
    # 8 minutes across 5 cameras). At 5s the loop spends at most ~1 read per
    # 5s of wall time (~12 reads/min/group when saturated), identification
    # still converges, and the cameras keep flipping slots. 0 disables pacing.
    slot_ocr_min_gap_s: float = 5.0

    # Run the slot-identify OCR read on a BACKGROUND thread instead of inline in the
    # serial camera loop. A PaddleOCR read is 2-8s on the production Xeon; inline, it
    # is the single largest per-frame cost and it freezes slot flips for EVERY camera
    # in the group while it runs. Off-thread, the loop submits a (crop, candidates)
    # job and keeps flipping slots at infer+zones speed; the read's result binds a
    # frame or two later, once the worker finishes.
    #
    # ONLY the paddle read moves — the ReID ranking and the classifier-based decision
    # log stay on the main thread (they share OpenVINO models with the tracking loop,
    # which is not safe to call from two threads). The binding itself also stays on
    # the main thread, and RE-VALIDATES the slot first: a read that lands after the
    # car has left, or after the slot was already named, is discarded, never bound.
    # So async cannot bind a stale plate; at worst it binds a few hundred ms later
    # than inline would have. False = the historical inline path (kept as a fallback).
    slot_ocr_async: bool = True

    # How many of the 12 identify attempts may pay for OCR's enlarged retry pass.
    # The retry rescues a plate that is present but too small for the text detector to
    # find (slot B13, CAM-24: '' -> '9990BHD' -> BHD-9990, a C-level slot dark for 60
    # attempts). On a slot whose plate is genuinely out of frame it instead surfaces
    # spurious text regions, runs the recogniser on them, discards them, and costs ~670ms
    # — on the frame loop, 12 times a park. If the plate were visible at all an early
    # frame would have caught it, so spend the budget there.
    ocr_upscale_retry_max_attempts: int = 4

    # --- Multi-frame OCR consensus (parked-slot identify) -----------------------
    # The single-frame slot-identify path binds only when appearance (ReID shortlist)
    # AND a plate read agree on the SAME frame. On a slot with no parked-pose gallery
    # reference (e.g. B13/CAM-24), ReID's cross-view score is inverted, so its shortlist
    # never contains the true car — and a PERFECT read is discarded because it "confirms
    # none of ReID's top-k". The pose is therefore never learned, so ReID never improves:
    # the OCR/ReID learning deadlock. This fallback breaks it WITHOUT loosening the
    # never-wrong contract: it accumulates reads across frames and binds a plate only when
    # the SAME inside-the-facility car (phantoms excluded) is the unique digit-run match on
    # >= min_agreement distinct reads, leading the runner-up by >= margin. Repeated reads
    # replace the appearance witness; each vote must still be an UNAMBIGUOUS single match,
    # so a digit-run collision (DJS-7842 vs phantoms) contributes nothing and the slot
    # stays NULL — correct-or-null preserved. Set enabled False for the historical
    # single-frame-only behaviour.
    slot_ocr_consensus_enabled: bool = True
    # Distinct agreeing reads before a consensus bind. 3 independent frames reading the
    # same >=4-digit run for the same inside car is ~1-in-10,000 per frame — far stronger
    # corroboration than the single-frame ReID+OCR path it falls back from.
    slot_ocr_consensus_min_agreement: int = 3
    # The winner must lead the second-most-voted plate by this many reads. Guards the case
    # where two inside cars share overlapping digit runs and reads split between them.
    slot_ocr_consensus_margin: int = 2

    # Decision log — one JSONL record per slot-identity attempt: the ranked candidates,
    # their full feature vectors, the gate rejects, the OCR read, and which candidate the
    # read confirmed (THE LABEL). Every bind yields 1 positive + up to k-1 hard negatives.
    #
    # This exists because a ranker trained offline on the gallery alone does NOT beat
    # plain ReID (measured 2026-07-13: cold rank-1 74.9% -> 71.1%, and the trainer refused
    # to ship it). The features that actually break ties — locked_elsewhere, the OCR read,
    # crop quality at that instant — only exist at decision time and cannot be
    # reconstructed later. So they are recorded as they happen.
    decision_log_enabled: bool = True
    decision_log_dir: str = "logs/decisions"
    decision_log_queue_max: int = 2000

    # A candidate VA has never photographed cannot be matched — it can only collide.
    # VA hydrates a session for every open parking_sessions row, and the ANPR gate
    # misreads: 2026-07-13, 18 of 51 "cars inside" had no gallery folder because they
    # are mis-OCR'd spellings of real plates (BJA-7842 / DJA-7842 next to the real
    # DJS-7842). confirm_plate() matches on the DIGIT RUN and abstains on ambiguity, so
    # those phantoms turn a perfect read of DJS-7842 into a 3-way tie -> slot stays NULL.
    # Require appearance evidence (a gallery reference) before a plate may be an OCR
    # candidate. The ReID path is already immune — a phantom has no vector, so it is
    # never scored — this closes the plates_inside() fallback.
    candidates_require_appearance_evidence: bool = True

    # A car cannot occupy two slots. Drop candidates already locked into a DIFFERENT
    # slot before the shortlist is truncated, so rank-6 promotes into the top-5 — the
    # gate can only ADD the right car to the candidate set, never remove it. Measured
    # 2026-07-13: this alone accounts for ~9 of the 34 cold rank-1 errors (TRS-9117 in
    # B5 kept losing to HSR-8327, which was already locked into B26).
    exclude_plates_locked_elsewhere: bool = True

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

    # Plate-lock bar: a voting-committed plate freezes onto a parked slot when
    # its ReID score clears this (OR OCR agreement, see engine_runtime). Defaults
    # to reid_solo_confirm; kept separate so the freeze bar can be tuned alone.
    lock_confidence: float = 0.70

    # Appearance floor for the B1 single-candidate fallback. When exactly one
    # provisional candidate remains but its ReID did not clear the confirm bar,
    # the historical policy confirmed it regardless of appearance — which let the
    # NEXT car adopt a PREVIOUS car's stale pending plate at ~0 ReID agreement
    # (the night gate identity-swap incident). The lone candidate is now confirmed
    # only when its ReID clears this floor (open-set novelty rejection). Default
    # 0.0 preserves legacy behaviour; production sets it in config.yaml. Calibrate
    # on the gate cross-view FAR/DIR curve — do NOT reuse b1_zone.
    single_candidate_min_reid: float = 0.0

    # Appearance-consistency (open-set novelty) floor for gallery admission. A new
    # reference crop is only merged into an identity's gallery when its ReID max-
    # similarity to that identity's ESTABLISHED refs clears this — so a foreign
    # car wrongly bound to a plate cannot poison that plate's gallery (the night
    # gate contamination: "the other car's images ended up in my gallery"). Also
    # makes the diversity-merge safe (a foreign vector is otherwise preferentially
    # kept by the farthest-point cap). Default 0.0 = off (legacy). Set
    # conservatively in config.yaml — reject only clearly-foreign crops (near-zero
    # agreement); tune on the FAR/DIR curve. Bootstrap (empty gallery) always admits.
    gallery_min_identity_similarity: float = 0.0

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
    # How many of the sharpest burst crops to keep as per-identity ReID
    # references when multishot is on (applies to every camera, not just the
    # entry camera). Matching scores max-cosine over these references.
    multishot_ref_top_k: int = 3

    # Minimum per-read OCR confidence to ACCEPT an ANPR entry read (source-side
    # gate on the two-plate genesis). A read below this — when the ANPR server
    # supplies a confidence — is held rather than minting a pending plate, so a
    # low-confidence night misread cannot create a second identity for one car.
    # Default 0.0 = accept all (and it is a no-op when the server omits
    # confidence). Set once the ANPR server sends per-read confidence; typical
    # ALPR practice gates around 0.8-0.9.
    anpr_min_accept_confidence: float = 0.0

    # Cross-session identity reconciliation: collapse two confirmed identities
    # that are the SAME physical car the ANPR misread as two plates. At confirm
    # time, a new session that has ReID similarity >= this floor to another
    # freshly-confirmed (not-yet-parked, different-plate) session entered within
    # identity_reconcile_window_seconds is treated as the same car and the
    # duplicate is closed. HIGH floor on purpose (well above the same-car mean
    # ~0.67) so only near-identical gate views trigger and two similar-but-
    # different cars are never merged. Default 0.0 = OFF — enabling closes real
    # sessions, so validate on multi-car footage first (recommended ~0.75 / 60s).
    identity_reconcile_min_similarity: float = 0.0
    identity_reconcile_window_seconds: float = 60.0

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
    # the specified directory, the matcher uses the OpenVINO runtime path
    # (target ≤40 ms/image on CPU). When the file is missing or the flag is off
    # the matcher falls back to the legacy torchreid path (~1 s/image on CPU).
    # ``reid_input_size`` is ``(height, width)`` and must agree with the size
    # baked into the exported IR.
    use_openvino_reid: bool = True
    reid_input_size: tuple = (256, 128)
    reid_openvino_model_dir: str = "models/PS_carMatching"

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
    enable_restricted_zone_alerts: bool = True

    # Named-slot intrusion is decided from IDENTITY, which lands well after the car
    # parks (OCR/ReID run after occupancy is published — see the identity block in
    # ParkingEngineRuntimeMixin._process_detections_and_events). So the ownership
    # verdict is deferred rather than guessed at park time.
    #
    # This is how long to wait for identity before giving up and raising a
    # reserved_slot_unidentified alert instead. Without it, a slot whose identity
    # NEVER resolves gets no alert at all — and that is not hypothetical: B1_CRO
    # has OCR disabled outright (matching.slot_no_plate_view) and can only ever be
    # named by appearance, which routinely abstains. That is a named slot with zero
    # intrusion coverage. Set <= 0 to disable the fallback and alert only on a
    # PROVEN non-owner.
    reserved_slot_identity_timeout_s: float = 300.0


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
        config.processing.max_grab_fps = int(p.get(
            "max_grab_fps", config.processing.max_grab_fps
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
        config.detector.ov_num_threads = int(
            d.get("ov_num_threads", config.detector.ov_num_threads) or 0
        )
        config.detector.ov_performance_hint = str(
            d.get("ov_performance_hint", config.detector.ov_performance_hint) or ""
        )
        config.detector.max_box_area_ratio = float(
            d.get("max_box_area_ratio", config.detector.max_box_area_ratio) or 0.0
        )
        config.detector.camera_overrides = _parse_detector_overrides(
            d.get("camera_overrides") or {}
        )

    # The supervisor's per-group thread cap. When each group is pinned to a
    # disjoint core slice, OpenVINO sizes its pool from the affinity mask and
    # this stays unset (see src/ov_tuning.py). But under a Kubernetes CPU *limit*
    # the mask is the whole NODE and pinning to host cores we don't own starves
    # whichever groups land on contended CPUs — so the supervisor unpins and caps
    # the pool explicitly instead. YAML/DB never carry this: it is a property of
    # how THIS process was launched, not of the deployment's tuning.
    _ov_env = os.environ.get("VA_OV_NUM_THREADS", "").strip()
    if _ov_env:
        try:
            config.detector.ov_num_threads = max(0, int(_ov_env))
        except ValueError:
            print(f"[WARN] VA_OV_NUM_THREADS={_ov_env!r} is not an int — ignored.")

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
            config.preprocessing.detector.detector_scale = d_pp.get(
                "detector_scale", config.preprocessing.detector.detector_scale
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

    # Alert toggles. Precedence: dataclass default -> YAML -> env var.
    #
    # This used to read the env var FIRST with a hardcoded "false" default, which
    # unconditionally clobbered the AlertsConfig default of True. Nothing in the repo
    # ever set that var, and no config.yaml ever carried an `alerts:` block — so from
    # a9ce2ee (2026-05-20) until this fix, EVERY restricted-zone alert (named-slot
    # intrusion, special-needs violation) was silently dead in every deployment.
    # The var is now an explicit deploy-time OVERRIDE: unset means "use the config",
    # not "off".
    if "alerts" in raw:
        a = raw["alerts"] or {}
        config.alerts.enable_restricted_zone_alerts = bool(
            a.get(
                "enable_restricted_zone_alerts",
                config.alerts.enable_restricted_zone_alerts,
            )
        )
        try:
            config.alerts.reserved_slot_identity_timeout_s = float(
                a.get(
                    "reserved_slot_identity_timeout_s",
                    config.alerts.reserved_slot_identity_timeout_s,
                )
            )
        except (TypeError, ValueError):
            pass
    _env_restricted_alerts = os.environ.get("ENABLE_RESTRICTED_ZONE_ALERTS")
    if _env_restricted_alerts is not None:
        config.alerts.enable_restricted_zone_alerts = (
            _env_restricted_alerts.strip().lower() == "true"
        )

    # alert_service reaches this flag through a module-level setter rather than
    # AppConfig: it is called from request handlers and DB paths that never see the
    # engine's config object. Without this the two gates desync — the engine would
    # tag the event vehicle_intrusion and save a snapshot while report_alert silently
    # dropped the row, producing alert LOGS with an empty alerts table.
    # Deferred + guarded: config loading must not become dependent on the ORM/DB
    # stack being importable (some tooling loads config standalone).
    try:
        from src.services import alert_service as _alert_service

        _alert_service.configure_alerts(
            enable_restricted_zone_alerts=config.alerts.enable_restricted_zone_alerts
        )
    except ImportError as exc:  # pragma: no cover - only in cut-down environments
        print(f"[WARN] Could not push alert config to alert_service: {exc}")

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
        if "reattach_excluded_cameras" in m:
            cm.reattach_excluded_cameras = tuple(
                m.get("reattach_excluded_cameras") or ()
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
        cm.lock_confidence = m.get("lock_confidence", cm.lock_confidence)
        cm.single_candidate_min_reid = m.get(
            "single_candidate_min_reid", cm.single_candidate_min_reid
        )
        cm.gallery_min_identity_similarity = m.get(
            "gallery_min_identity_similarity", cm.gallery_min_identity_similarity
        )
        cm.global_match_margin = m.get(
            "global_match_margin", cm.global_match_margin
        )

        # Per-car persistent gallery
        cm.gallery_persist_enabled = m.get(
            "gallery_persist_enabled", cm.gallery_persist_enabled
        )
        cm.gallery_max_refs_per_car = m.get(
            "gallery_max_refs_per_car", cm.gallery_max_refs_per_car
        )
        cm.gallery_retention_days = m.get(
            "gallery_retention_days", cm.gallery_retention_days
        )
        cm.gallery_min_view_quality = m.get(
            "gallery_min_view_quality", cm.gallery_min_view_quality
        )
        cm.gallery_neighbour_clearance_enforce = bool(
            m.get(
                "gallery_neighbour_clearance_enforce",
                cm.gallery_neighbour_clearance_enforce,
            )
        )
        cm.gallery_require_slot_authority = bool(
            m.get(
                "gallery_require_slot_authority",
                cm.gallery_require_slot_authority,
            )
        )
        cm.slot_plate_requires_ocr = bool(
            m.get("slot_plate_requires_ocr", cm.slot_plate_requires_ocr)
        )
        cm.slot_acquire_by_ocr_transit = bool(
            m.get("slot_acquire_by_ocr_transit", cm.slot_acquire_by_ocr_transit)
        )
        cm.ocr_transit_window_seconds = float(
            m.get("ocr_transit_window_seconds", cm.ocr_transit_window_seconds)
        )
        cm.slot_acquire_by_elimination = bool(
            m.get("slot_acquire_by_elimination", cm.slot_acquire_by_elimination)
        )
        cm.slot_acquire_min_similarity = float(
            m.get("slot_acquire_min_similarity", cm.slot_acquire_min_similarity)
        )
        cm.slot_acquire_inflight_seconds = float(
            m.get("slot_acquire_inflight_seconds", cm.slot_acquire_inflight_seconds)
        )
        cm.gallery_min_sharpness = m.get(
            "gallery_min_sharpness", cm.gallery_min_sharpness
        )
        cm.gallery_dedup_cosine = m.get(
            "gallery_dedup_cosine", cm.gallery_dedup_cosine
        )
        cm.gallery_accumulate_interval_s = m.get(
            "gallery_accumulate_interval_s", cm.gallery_accumulate_interval_s
        )
        cm.gallery_accumulate_min_gt_similarity = m.get(
            "gallery_accumulate_min_gt_similarity",
            cm.gallery_accumulate_min_gt_similarity,
        )
        cm.gallery_min_crop_area = m.get(
            "gallery_min_crop_area", cm.gallery_min_crop_area
        )
        cm.confirmation_extra_rear_views = m.get(
            "confirmation_extra_rear_views", cm.confirmation_extra_rear_views
        )
        if "ground_truth_cameras" in m:
            cm.ground_truth_cameras = tuple(m.get("ground_truth_cameras") or ())
        cm.secondary_camera_weight = m.get(
            "secondary_camera_weight", cm.secondary_camera_weight
        )
        cm.slot_camera_ref_weight = m.get(
            "slot_camera_ref_weight", cm.slot_camera_ref_weight
        )
        cm.exclude_plates_locked_elsewhere = m.get(
            "exclude_plates_locked_elsewhere", cm.exclude_plates_locked_elsewhere
        )
        cm.candidates_require_appearance_evidence = m.get(
            "candidates_require_appearance_evidence",
            cm.candidates_require_appearance_evidence,
        )
        cm.ocr_upscale_retry_max_attempts = m.get(
            "ocr_upscale_retry_max_attempts", cm.ocr_upscale_retry_max_attempts
        )
        cm.slot_reid_solo_enabled = m.get(
            "slot_reid_solo_enabled", cm.slot_reid_solo_enabled
        )
        cm.slot_reid_solo_after_attempts = m.get(
            "slot_reid_solo_after_attempts", cm.slot_reid_solo_after_attempts
        )
        cm.slot_reid_solo_min_score = m.get(
            "slot_reid_solo_min_score", cm.slot_reid_solo_min_score
        )
        cm.slot_reid_solo_min_margin = m.get(
            "slot_reid_solo_min_margin", cm.slot_reid_solo_min_margin
        )
        cm.slot_reid_solo_min_score_reserved = m.get(
            "slot_reid_solo_min_score_reserved", cm.slot_reid_solo_min_score_reserved
        )
        cm.slot_reid_solo_min_margin_reserved = m.get(
            "slot_reid_solo_min_margin_reserved", cm.slot_reid_solo_min_margin_reserved
        )
        cm.slot_reid_retry_interval_s = float(
            m.get("slot_reid_retry_interval_s", cm.slot_reid_retry_interval_s) or 0.0
        )
        cm.slot_ocr_min_gap_s = float(
            m.get("slot_ocr_min_gap_s", cm.slot_ocr_min_gap_s) or 0.0
        )
        if "slot_ocr_async" in m:
            cm.slot_ocr_async = bool(m.get("slot_ocr_async"))
        if "slot_ocr_consensus_enabled" in m:
            cm.slot_ocr_consensus_enabled = bool(m.get("slot_ocr_consensus_enabled"))
        cm.slot_ocr_consensus_min_agreement = int(
            m.get("slot_ocr_consensus_min_agreement", cm.slot_ocr_consensus_min_agreement)
        )
        cm.slot_ocr_consensus_margin = int(
            m.get("slot_ocr_consensus_margin", cm.slot_ocr_consensus_margin)
        )
        # Normalised once, here, so every lookup downstream is a plain set membership
        # on the same spelling the DB/API use for slot_id.
        cm.slot_no_plate_view = [
            str(s).strip()
            for s in (m.get("slot_no_plate_view", cm.slot_no_plate_view) or [])
            if str(s).strip()
        ]
        cm.decision_log_enabled = m.get("decision_log_enabled", cm.decision_log_enabled)
        cm.decision_log_dir = m.get("decision_log_dir", cm.decision_log_dir)
        cm.decision_log_queue_max = m.get(
            "decision_log_queue_max", cm.decision_log_queue_max
        )

        # HSV tolerances
        cm.hsv_h_tol = m.get("hsv_h_tol", cm.hsv_h_tol)
        cm.hsv_s_tol = m.get("hsv_s_tol", cm.hsv_s_tol)
        cm.hsv_v_tol = m.get("hsv_v_tol", cm.hsv_v_tol)

        # Feature flags — yaml overrides env-var defaults set above
        cm.use_color_filter = m.get("use_color_filter", cm.use_color_filter)
        cm.use_lab_clahe = m.get("use_lab_clahe", cm.use_lab_clahe)
        cm.use_multishot = m.get("use_multishot", cm.use_multishot)
        cm.multishot_ref_top_k = m.get("multishot_ref_top_k", cm.multishot_ref_top_k)
        cm.anpr_min_accept_confidence = m.get(
            "anpr_min_accept_confidence", cm.anpr_min_accept_confidence
        )
        cm.identity_reconcile_min_similarity = m.get(
            "identity_reconcile_min_similarity", cm.identity_reconcile_min_similarity
        )
        cm.identity_reconcile_window_seconds = m.get(
            "identity_reconcile_window_seconds", cm.identity_reconcile_window_seconds
        )

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
