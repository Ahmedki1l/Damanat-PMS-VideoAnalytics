# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies FIRST (important ترتيب)
RUN pip install --no-cache-dir "numpy<2.0.0"
RUN pip install --no-cache-dir Cython scipy
RUN pip install --no-cache-dir opencv-python-headless

# Install PyTorch CPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir torchreid==0.2.5

# Install build-heavy dependencies separately to cache layers
RUN pip install --no-cache-dir "lap>=0.5.12" "openvino>=2024.0.0"

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt


# =========================
# Stage 2: Runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Plate-OCR weights, baked. PaddleOCR otherwise downloads det/rec/angle (~20MB)
# into $HOME/.paddlex on the FIRST read() — which in a pod means an internet
# egress the cluster may not allow, into a $HOME that may not be writable (see
# the Ultralytics '/root/.config not writable' warning). Both failure modes hit
# mid-run, long after the pod looks healthy, and land in a blanket `except` in
# vehicle_registry.py that silently degrades to NoopPlateOCR — i.e. plates just
# stop binding, with no crash to point at.
#
# So: point PaddleX's cache at a fixed dir and populate it at BUILD time. The
# warm-up must construct PaddleOCR with the same model names + angle flag that
# PaddlePlateOCR._ensure_engine() uses, or it would cache the wrong weights.
# The supervisor copies its env to all 5 groups, so one ENV covers every worker.
ENV PADDLE_PDX_CACHE_HOME=/opt/paddlex
RUN python -c 'import numpy as np; from paddleocr import PaddleOCR; PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True, enable_mkldnn=False, device="cpu", text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="en_PP-OCRv5_mobile_rec").predict(np.zeros((320, 320, 3), dtype=np.uint8))' \
    && ls /opt/paddlex/official_models

# Line-buffer stdout: docker hands the process a pipe, not a tty, and Python
# block-buffers 8KB to a pipe — so without this the `[INFO] effective FPS/camera`
# and `[PERF]` lines only reach `docker logs` in delayed bursts.
ENV PYTHONUNBUFFERED=1

# Per-stage timing plus RTSP-drain/frame-publication meters is
# opt-in — flip it on at run time, no rebuild:
#   docker run -e PERF_TRACE=1 -e PERF_TRACE_EVERY=50 ...
# The effective-FPS summary is always on and needs nothing.
# ============================================================================
# >>> MEASUREMENT BUILDS — edit this block, rebuild, deploy, grab the log.
# PROD = all overrides empty, PERF_TRACE=0 (restore when done).
#
#   BUILD 1  Isolated b1 (uncontended `ov` floor). Runs ONLY b1 — other floors
#            stop updating while deployed; keep it brief.
#              PERF_TRACE=1  VA_OV_NUM_THREADS=7
#              VA_CMD="python main.py --cameras CAM-04,CAM-05,CAM-06,CAM-07,CAM-08,CAM-20,CAM-21,CAM-22,CAM-24"
#            RESULT: ov ~24ms, ~2.2 fps/cam (floor).
#
#   BUILD 2  Full workload, thread cap → 2/2/5/6.  PERF_TRACE=1 VA_OV_TOTAL_THREADS=15
#            RESULT: ov ~123ms (~10% only) — thread count is NOT the lever.
#
#   BUILD 3  Full workload, MERGE to 2 processes (2 OpenVINO pools instead of 4),
#            same 15-thread total. Tests whether the residual gap is PROCESS-level
#            contention. No VA_CMD; supervisor runs normally. Groups become
#            gate+b2 (8 thr, api) and b1+ground (7 thr) — read b1's numbers from
#            the [b1+ground] infer-breakdown line.
#              PERF_TRACE=1  VA_OV_TOTAL_THREADS=15  VA_MERGE_GROUPS=2
#            Note: measurement topology only (re-buckets cameras across floors) —
#            revert after. If ov moves toward ~24ms, process contention is dominant.
#
#   BUILD 4  *** PHASE 2 — READY FOR PRODUCTION VALIDATION (not yet production-ready). ***
#            ONE process feeds every camera through ONE OpenVINO AsyncInferQueue
#            (THROUGHPUT). Bypasses the supervisor entirely (VA_CMD runs main.py
#            directly, with --api for the ANPR webhook + SSE). VA_SINGLE_PROCESS=1
#            forces the async detector core. Set VA_OV_NUM_THREADS to the pod's real
#            core count (the single pool). Parity + concurrency are dev-box proven;
#            the live 26-camera load and the real ReID/registry path are the unknowns
#            this build exists to measure.
#              PERF_TRACE=1  VA_SINGLE_PROCESS=1  VA_INFER=async
#              VA_OV_NUM_THREADS=15  VA_CMD="python main.py --api"
#            HARD GATE (design doc): ov must drop toward ~24-40ms and per-camera fps
#            must rise. Watch [PERF] infer-breakdown + [PERF] per-camera-fps, and
#            [PERF] consumer busy% — if ov drops but fps stays capped AND consumer
#            busy% is near 100, the bottleneck moved to the inline ReID/DB/snapshots
#            (Phase 3/4 targets). If it regresses, revert to PROD (multi-process) —
#            do NOT proceed to Phase 3-5.
#
#   PROD     PERF_TRACE=0, all overrides empty (multi-process supervisor).
#
# Currently set to: BUILD 5 (gate-camera capacity isolation, multi-process
# supervisor). BUILD 4's single-process overrides are cleared below.
# To go back to BUILD 4: restore the VA_SINGLE_PROCESS/VA_INFER/VA_CMD/
# VA_OV_NUM_THREADS/VA_FEED_THREADS/VA_INFER_NIREQ values listed there.
# To reach true PROD: PERF_TRACE=0 as well.
# ============================================================================
# PROD value. Was 1 for the entry-v3 shadow window, where the per-stage timing
# was the point; it must not ride into the authoritative flip, because the
# fleet is already CPU-starved (measured 0.1 fps/cam) and this pays for timing
# on every stage of every frame. Flip it back at run time when measuring —
# `-e PERF_TRACE=1` — rather than rebuilding.
ENV PERF_TRACE=0
ENV PERF_TRACE_EVERY=50
#
#   BUILD 5  *** CURRENT — gate-camera capacity ISOLATION. ***
#            BUILD 4 put all 27 cameras on ONE AsyncInferQueue, which makes the
#            entry ramp's frame rate a RESIDUAL of whatever the parking cameras
#            leave behind. Measured: CAM-23 ran 0.10 fps until motion gating was
#            promoted to enforce, then 1.26 — but that entire gain came from
#            SUPPRESSING QUIET cameras, so it evaporates exactly when the garage
#            is busy and entries are most frequent. Raising CAM-23 to 2688x1520
#            alone dropped it to 0.90 and total inference 14.6 -> 8.0 inf/s.
#            A shared queue cannot give the entry path a floor.
#
#            The supervisor already models the fix: _groups_from_db() pulls
#            VA_GATE_CAMERAS (CAM-23, CAM-03) OUT of their area buckets into a
#            dedicated "gate" group, so they get their own process and their own
#            OpenVINO pool. Parking activity can no longer starve them.
#
#            So BUILD 4's overrides are cleared and the default supervisor
#            ENTRYPOINT runs. VA_MERGE_GROUPS stays EMPTY on purpose: merging to
#            2 groups folds gate together with b2 (see BUILD 3) and destroys the
#            isolation this build exists to create.
#
#            HARD GATE: compare gate-group fps QUIET vs BUSY. Holding its rate
#            under load is the whole point. Watch for the 2026-07-15 regression
#            in supervisor.py (~line 311) — too many groups apportioned gate ONE
#            thread and single-threaded HEVC decode floored it at 0.40 fps;
#            _MIN_THREADS_PER_GROUP=2 guards that, but 2 threads may still be
#            thin for a 2688x1520 stream. If gate sits near 0.4, that is thread
#            starvation, not the queue. Revert = restore the BUILD 4 block below.
# ---------------------------------------------------------------------------
# BUILD 4 (single-process async) — kept for one-line revert:
#   VA_INFER_NIREQ=16  VA_SINGLE_PROCESS=1  VA_INFER=async
#   VA_OV_NUM_THREADS=16  VA_FEED_THREADS=6  VA_CMD="python main.py --api"
ENV VA_INFER_NIREQ=""
ENV VA_SINGLE_PROCESS=""
ENV VA_INFER=""
ENV VA_OV_NUM_THREADS=""
ENV VA_FEED_THREADS=""
ENV VA_CMD=""
# The two cameras that must not compete with parking for inference: the ramp-top
# line crossing and the B1 entrance backstop. supervisor.py forces them into the
# single --api group, which it must be anyway — the ANPR handler reads their live
# frames through a PROCESS-LOCAL buffer, so co-location is required, not optional.
ENV VA_GATE_CAMERAS="CAM-23,CAM-03"
# Safe rollout defaults. Deployment-time env may promote Entry V2 from off to
# shadow/authoritative. CREDENTIALS are never baked; site calibration IS baked
# below, as a sane fallback rather than as the source of truth.
#
# PRECEDENCE, and it has bitten this deployment twice: in Kubernetes an
# `envFrom.secretRef` value OVERRIDES the image ENV. Everything below is a
# DEFAULT that the deployed Secret wins against. That is deliberate — the Secret
# is where a value gets swept without a rebuild — but it means a stale key in the
# Secret silently defeats a change made here. After any rollout that changes one
# of these, verify against the RUNNING pod, never against this file:
#
#   kubectl exec deploy/pms-video-analytics -- printenv | grep ^ENTRY_V2_
#
# These are baked because `kubectl create secret --from-env-file` REPLACES the
# whole Secret, which has twice dropped the camera-policy variables entirely. An
# image default turns that from a silent policy change into a survivable one.
ENV ENTRY_V2_MODE=off
# NOTE: this pins the variable that OVERRIDES config.yaml's motion_scheduler.mode
# (config.py reads VA_MOTION_SCHEDULER_MODE last and it wins).
#
# MUST be legacy for BUILD 5. Motion scheduling exists only in the single-process
# async engine, and main.py fails closed (exit 2) on shadow/enforce unless
# VA_SINGLE_PROCESS=1 — which BUILD 5 clears above. Left at shadow it took prod
# down: every supervisor group exited 2 at startup ("motion scheduler
# shadow/enforce modes require VA_SINGLE_PROCESS=1 in multi-camera mode"), the
# supervisor tore down the rest, CrashLoopBackOff. The BUILD 4 -> BUILD 5 revert
# cleared VA_SINGLE_PROCESS but left this at shadow (2026-07-28).
#
# Costs the gate nothing: CAM-23/CAM-03 bypass gating anyway
# (motion_scheduler.camera_overrides). Enforce helped them only indirectly, by
# suppressing the other 25 cameras on BUILD 4's shared queue (~0.10 -> ~2.1 fps)
# — a gain that evaporated under load. BUILD 5's per-group OpenVINO pool is the
# replacement. Restore shadow/enforce only with the BUILD 4 block.
ENV VA_MOTION_SCHEDULER_MODE=legacy
ENV VA_SLOT_STATE_MODE=shadow
# Field-calibration diagnostic: persist each zone ENTRY frame to
# /app/vehicle_images/entry_zone_captures/. Zone crops are RAM-only, and a visit
# discarded for tracker_loss/ambiguity never reaches the analyzer, so no image is
# written anywhere and "was the car captured, and is the plate legible?" cannot be
# answered from disk. /app/vehicle_images is the durable gallery mount, so the
# files are readable from the host and survive restarts.
#
# Set EMPTY to disable — it persists transit imagery of every vehicle entering
# the zone, so it belongs on for a calibration drive and off afterwards:
#   docker run -e ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR= ...
ENV ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR=/app/vehicle_images
# --- Supervisor-only knobs (ignored while VA_CMD bypasses the supervisor) ---
ENV VA_OV_TOTAL_THREADS=""
ENV VA_MERGE_GROUPS=""

# ============================================================================
# Entry Pipeline v3 — shadow-window defaults
# ============================================================================

# Emit logger.info at all. Without this the root logger has no handler, records
# fall through to logging.lastResort (WARNING), and every logger.info() in src/
# is discarded — including the [EntryV2][ReID] lines the calibration depends on.
ENV LOG_LEVEL=INFO
# Per-logger overrides, "logger=LEVEL,logger=LEVEL". EMPTY is correct: the
# per-frame caps are applied in CODE (logging_setup.py:67-68 lists the noisy
# loggers, :139 pins them to WARNING), so raising the root logger to INFO does
# not switch on engine_tracking's per-detection "[quality] track=..." line on an
# already CPU-starved fleet. Set one here only to debug it deliberately:
#   LOG_LEVEL_OVERRIDES=src.core.engine.engine_tracking=INFO
ENV LOG_LEVEL_OVERRIDES=

# The pipeline raises NO alerts — every outcome is a record here, so this
# directory is the entire operational surface. Must be on the PVC-backed gallery
# mount: every record cites image paths on that volume, and a log that outlives
# the images it references is not evidence. Empty disables it.
ENV ENTRY_V2_DECISION_LOG_DIR=/app/vehicle_images/entry_v2_shadow
# Rolls over daily, never prunes by mtime — files are deleted by the DAY IN
# THEIR FILENAME. It shares a volume with vehicle imagery.
ENV ENTRY_V2_DECISION_LOG_RETENTION_DAYS=30

# How long an UNCONFIRMED identity stays eligible for correlation, and how long
# an unmatched observation outlives it so a late HikCentral sweep can still
# rescue a dropped entry. Observation TTL must be >= identity TTL or config
# fails. This is 90x the legacy path's 10s FIFO bind window, and safe ONLY
# because nothing here binds a plate by arrival order — if FIFO ever returns to
# the binding path, bring these down with it.
ENV ENTRY_IDENTITY_TTL_MINUTES=15
ENV ENTRY_OBSERVATION_TTL_MINUTES=60

# Colour REMOVES a candidate that cannot be this car; the margin is then
# recomputed over the survivors. It never adds score — two white sedans agreeing
# on colour is not evidence they are one car. Fails open on missing colour.
# Vetoed nothing incorrectly across the 2026-08-30/31 window.
ENV ENTRY_V2_COLOUR_VETO_ENABLED=1

# OFF, measured — ramp OCR is not reliable enough to WITHHOLD an entry on.
# Two reads of the SAME car in one window, both above the 0.75 confidence gate:
# "7383HAS" and "AATEIGH7383HAS". The veto compares exact plate keys
# (decision.py:500 `item.key != key`), so a hallucinated prefix reads as a
# contradiction — as does the known letters-first vs digits-first swap, e.g. a
# ramp read of "7286EED" against a consensus plate of "EED7286". Both withhold a
# CORRECT entry. The same exact-key flaw sits in the producer-family gate
# (coordinator.py:2181), which refused one real crossing 13 times on the 30th.
# Re-enable only behind a digit-run comparison, never behind exact keys.
ENV ENTRY_V2_OBSERVATION_PLATE_VETO_ENABLED=0

# Normalised (x1,y1,x2,y2) boxes in 0..1, semicolon-separated, naming where
# Hikvision composites its own plate/OSD panel into a frame. LEAVE EMPTY until
# the panel geometry is measured against this facility's own frames — a guessed
# rectangle rejects real plates, which is worse than the echo it prevents.
ENV ENTRY_V2_OVERLAY_EXCLUDE_REGIONS=

# Camera crossing policy. The line/direction values are the RAW Hikvision
# strings; they must not be quoted here or in the Secret — a literal quote
# survives into the parsed set ("'1", "PARK_ENTRY'") and every crossing on that
# camera is then rejected as crossing_line_not_configured_for_role, invisibly,
# because configuration_errors() short-circuits while the mode is off.
ENV ENTRY_V2_PRIMARY_CAMERAS=CAM-23
ENV ENTRY_V2_PRIMARY_LINES=1,Park_Entry
ENV ENTRY_V2_PRIMARY_DIRECTIONS=ramp-entry
ENV ENTRY_V2_FALLBACK_CAMERAS=CAM-03
ENV ENTRY_V2_FALLBACK_LINES=1,B1_Entrence
ENV ENTRY_V2_FALLBACK_DIRECTIONS=B-to-A,b-entry

# Measured, 2026-08-30/31 shadow window (560 records, 57 identities, 66 ramp
# observations). The shipped 0.75 was UNREACHABLE: the maximum ANPR->ramp Re-ID
# score observed was 0.689, so zero entries could ever confirm — 55 identities
# expired holding only ['anpr'] and 65 observations expired with no witness.
#
# Per-observation best score is near-identical on the two cameras (CAM-23 median
# 0.406, CAM-03 0.402), so ONE threshold is right here. The per-RECORD split
# that suggests otherwise (0.417 vs 0.197) is an artifact: the 365 evaluations
# cover only 66 observations, and CAM-03 is re-scored against more wrong
# identities.
#
#   min_score   confirmable/57    both-ramp pairs
#     0.30        31 (54%)             14
#     0.35        26 (46%)             13   <- top of the plateau
#     0.40        22 (39%)             11
#     0.45        17 (30%)              4   <- cliff
#     0.75         0 ( 0%)              0   <- as shipped
#
# Margins carry the discrimination, not absolute score: 48 of 53 contested
# observations clear 0.08, median row margin 0.21-0.28. That matches the
# ReID-alone precision curve measured on the 50-car gallery, where margin — not
# score — separated the classes. If this value moves, keep the margins at 0.08+.
#
# NOT a precision measurement. The only truth signal in that window was 8 ramp
# plate reads, all of which CORROBORATED the Re-ID argmax (down to score 0.18)
# with zero contradictions. Eight is a signal, not a calibration; the window
# that runs at 0.35 is the one that measures whether these confirmations are
# actually right.
#
# ── That window ran. 2026-09-02, 28 real entries, ground truth = 27 CAM-23 ramp
# crossings (CAM-03 corroborates at 26) plus one HikCentral-only entry.
#
# Precision at 0.35 is 19/19 — every confirmation was a real car, no phantom.
# So the answer to the question above is yes, and the plateau extends further
# down than the 2026-08-30/31 table could see:
#
#   floor  booked  real  PHANTOM   recovered
#   0.35     19     19      0      <- previous setting
#   0.30     22     22      0      HER-9235, EEB-80, RLR-2714
#   0.20     25     25      0      + TXR-4857, ZZR-1372, RGR-6466
#   0.15     25     25      0      no further gain
#
# It plateaus at 0.20 because the MARGIN gate becomes the binding constraint —
# which is the note above, confirmed: 6 of the 8 refusals that day were blocked
# by score alone with a completely clean margin, and only the two RGR twins were
# refused for real ambiguity. In 51 of 85 evaluations row_margin EQUALS score
# (no runner-up at all), so below the contested cases the score gate was not a
# second opinion — it was the margin gate again, set 15 points too high.
#
# Hence: score demoted to a floor against unusable crops, margin left to decide.
# 0.20 also sits just above the 0.18 the ramp plate reads corroborated to.
# KEEP THE MARGINS AT 0.08 — they are now the only thing separating a car from
# its own misread twin (RGR-6466 clears at 0.086, RGR-6666 tops out at 0.020).
#
# Still shadow-only. Do not promote on this: at 25/28 it books fewer cars than
# the live path's 24-correct-plus-2-phantom, so a cutover today trades 2 phantoms
# for 3 missing cars. Confirm the plateau holds over a full week first.
ENV ENTRY_V2_REID_MIN_SCORE=0.20

# These are the names the code actually reads (settings.py:308-309). The single
# ENTRY_V2_REID_MIN_MARGIN that used to sit here was read by NOTHING — grep the
# tree and the Dockerfile is its only occurrence. Both margins happened to be
# right anyway, because 0.08 is also the code default, so the mistake was
# invisible: the block above says "KEEP THE MARGINS AT 0.08" and pointed at a
# variable that could not have moved them either way.
ENV ENTRY_V2_REID_ROW_MARGIN=0.08
ENV ENTRY_V2_REID_COLUMN_MARGIN=0.08

# ── Durable gallery as a Re-ID reference set ────────────────────────────────
# NOT the ENTRY_V2_GALLERY_REID_* family, which is the ADMISSION policy for
# writing to the gallery (min score 0.85, margins 0.12). These two are about
# READING it. The prefix is MATCH for exactly that reason — the two were one
# namespace for an afternoon and it was already confusing.
#
# WHY. Until 2026-09-06 src/entry never opened the gallery at all: a CAM-23
# crossing was scored only against this visit's CAM-ENTRY gate crops. That is
# the hardest comparison the system can make — cross-view, two reference
# vectors — and it is why the accepted band tops out at 0.819 with a median of
# 0.66. Meanwhile the store keeps precisely what this query wants: load_vectors
# EXCLUDES gate crops from matching and returns CAM-23/CAM-03/parked views,
# pruned by select_diverse_indices for viewpoint coverage. On 2026-09-06 all 27
# confirmed cars already had a gallery, 23 of them a full twenty references,
# and not one was consulted.
#
# READ-ONLY. Entry borrows the gallery; it does not teach it. Unsetting this
# restores the previous matcher exactly, with nothing to undo on disk — which
# is why it is safe to bake rather than to hold back for a Secret.
#
# VERIFIED before shipping: with this OFF, all 1007 recorded Re-ID decisions
# from the 2026-08-30..09-06 corpus replay identically through the new engine
# (tools/replay_entry_decisions.py --verify). The flag is the only thing that
# changes behaviour.
ENV ENTRY_V2_GALLERY_MATCH_ENABLED=1
# The SAME cap for every candidate, selected for viewpoint diversity rather
# than sharpness. max_similarity over more vectors is biased upward, so an
# uncapped comparison would let a twenty-reference car out-margin a
# two-reference one on sample count alone — which is a statement about the
# gallery, not about either car. Raise only with a margin recalibration.
ENV ENTRY_V2_GALLERY_MATCH_MAX_REFS=8

# Retire a same-plate identity that a confirmed one has superseded IN SOURCE
# TIME. The gate reads a plate more than once per car; when two reads are too
# dissimilar to merge at ENTRY_V2_MERGE_MIN_SCORE they become two identities
# under one key, one takes the ramp crossing, and the leftover stays a live
# Re-ID competitor for its full ENTRY_IDENTITY_TTL_MINUTES. On 2026-09-06 that
# leftover won a DIFFERENT car's crossing: GGR-9064 read at 10:09:18 and
# 10:10:34, confirmed on the 10:11:20 crossing, then its duplicate took
# RGR-6466's 10:24:04 CAM-23 crossing at 0.636 against RGR-6466's own 0.388.
# One car, two witnesses, two identities, both confirmed.
#
# Source time is the whole test, never plate equality: two live identities may
# legitimately share a plate — that is what an exit-and-re-entry IS — and a
# re-entry's gate read necessarily FOLLOWS the previous crossing, so it is
# untouched. Default ON in code; pinned here so the value is visible in
# `printenv` alongside everything else it interacts with.
ENV ENTRY_V2_SAME_KEY_RETIREMENT_ENABLED=1

# Copy app code
COPY . .

EXPOSE 8000

# The default exec is the Python supervisor (`--supervise --foreground`): it runs
# as PID 1 and spawns/supervises the 5 camera groups, mirrors each group's logs to
# stdout, and forwards SIGTERM on `docker stop`. VA_CMD overrides it (BUILD 4 runs
# one `main.py --api` instead). Extra launcher flags via RUN_ALL_ARGS, e.g.
# -e RUN_ALL_ARGS="--reset-plates".
#
# Zoning geometry is NOT seeded here. The DB `parking_slots`/`boundaries`/
# `parking_areas` tables are authoritative and long since populated; re-seeding from
# a checked-in dump on every boot could only ever reintroduce stale polygons. Seed a
# fresh database by hand — `python tools/sync_geometry.py seed --in geometry.json`
# (see README) — which is the only situation that ever needed it.
ENTRYPOINT ["sh", "-c", "exec ${VA_CMD:-python main.py --supervise --foreground ${RUN_ALL_ARGS:-}}"]
