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

# 🔥 IMPORTANT: استخدم نسخة stable بدل Git
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
# Currently set to: BUILD 4 (Phase 2 single-process async).
# To revert to prod multi-process: set VA_SINGLE_PROCESS/VA_INFER/VA_CMD empty,
# PERF_TRACE=0, and restore VA_OV_TOTAL_THREADS/VA_MERGE_GROUPS as needed.
# ============================================================================
ENV PERF_TRACE=1
ENV PERF_TRACE_EVERY=50
# --- Phase 2 (BUILD 4) single-process async engine -------------------------
# One process, one AsyncInferQueue, all cameras. Bypasses the supervisor.
# 1st prod run (8->10->9->11) showed the pool STARVED (~2/15 in flight) while the
# consumer + decode were idle: the single scheduler thread doing all preprocess
# was the ceiling (~20 inf/s). VA_FEED_THREADS spreads preprocess across N feeder
# threads (OpenCV/numpy release the GIL) so the pool fills. Watch [PERF] async-infer
# `~in flight` rise from ~2 toward nireq, and inf/s climb from ~20. If req_wall
# INFLATES and concurrency stays low, feeders+15 threads oversubscribe → next build
# lowers VA_OV_NUM_THREADS (~10) to give feeders CPU.
ENV VA_INFER_NIREQ=16
ENV VA_SINGLE_PROCESS=1
ENV VA_INFER=async
ENV VA_OV_NUM_THREADS=16
ENV VA_FEED_THREADS=6
ENV VA_CMD="python main.py --api"
# Safe rollout defaults. Deployment-time env may promote Entry V2 from off to
# shadow/authoritative, but credentials and site calibration are never baked.
ENV ENTRY_V2_MODE=off
ENV VA_MOTION_SCHEDULER_MODE=shadow
ENV VA_SLOT_STATE_MODE=shadow
# --- Supervisor-only knobs (ignored while VA_CMD bypasses the supervisor) ---
ENV VA_OV_TOTAL_THREADS=""
ENV VA_MERGE_GROUPS=""

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
