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

# Per-stage timing (roi/clahe/infer/zones/assign/slot + the decode meter) is
# opt-in — flip it on at run time, no rebuild:
#   docker run -e PERF_TRACE=1 -e PERF_TRACE_EVERY=50 ...
# The effective-FPS summary is always on and needs nothing.
# ============================================================================
# >>> MEASUREMENT BUILDS — edit this block, rebuild, deploy, grab the log.
# Three states. PROD = all four lines below empty/off (restore when done).
#
#   BUILD 1  Isolated b1 (uncontended `ov` floor). Runs ONLY b1 — other floors
#            stop updating while deployed; keep it brief.
#              PERF_TRACE=1  VA_OV_NUM_THREADS=7  SEED_GEOMETRY_ON_START=false
#              VA_CMD="python main.py --cameras CAM-04,CAM-05,CAM-06,CAM-07,CAM-08,CAM-20,CAM-21,CAM-22,CAM-24"
#
#   BUILD 2  Fix-B under FULL workload (all groups run; prod stays up). Caps
#            total OpenVINO threads to the real core count so the per-group split
#            becomes 2/2/5/6 instead of 2/2/7/8 — no VA_CMD, so the supervisor
#            runs normally.
#              PERF_TRACE=1  VA_OV_TOTAL_THREADS=15   (VA_CMD empty)
#
#   PROD     PERF_TRACE=0, all overrides empty.
#
# Currently set to: BUILD 1 (isolated b1).
# ============================================================================
ENV PERF_TRACE=1
ENV PERF_TRACE_EVERY=50
ENV SEED_GEOMETRY_ON_START=false
# --- BUILD 1 knobs (isolated b1) ---
ENV VA_OV_NUM_THREADS=7
ENV VA_CMD="python main.py --cameras CAM-04,CAM-05,CAM-06,CAM-07,CAM-08,CAM-20,CAM-21,CAM-22,CAM-24"
# --- BUILD 2 knob (full workload, capped threads → 2/2/5/6). Set for build 2,
#     and clear VA_CMD + VA_OV_NUM_THREADS above. ---
ENV VA_OV_TOTAL_THREADS=""

# Copy app code
COPY . .

EXPOSE 8000

# Same geometry seed as the single-process image; the only difference is the
# final exec — the Python supervisor (`--supervise --foreground`) runs as PID 1
# and spawns/supervises the 5 camera groups instead of one `main.py --api`. It
# mirrors each group's logs to stdout and forwards SIGTERM on `docker stop`.
# Extra launcher flags via RUN_ALL_ARGS, e.g. -e RUN_ALL_ARGS="--reset-plates".
ENTRYPOINT ["sh", "-c", "if [ \"${SEED_GEOMETRY_ON_START:-true}\" = true ] && [ -f \"${GEOMETRY_FILE:-geometry.json}\" ]; then echo '[entrypoint] seeding geometry into empty tables...'; python tools/sync_geometry.py seed --in \"${GEOMETRY_FILE:-geometry.json}\" --if-empty || echo '[entrypoint] geometry seed skipped (continuing)'; fi; exec ${VA_CMD:-python main.py --supervise --foreground ${RUN_ALL_ARGS:-}}"]
