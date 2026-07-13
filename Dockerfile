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
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Line-buffer stdout: docker hands the process a pipe, not a tty, and Python
# block-buffers 8KB to a pipe — so without this the `[INFO] effective FPS/camera`
# and `[PERF]` lines only reach `docker logs` in delayed bursts.
ENV PYTHONUNBUFFERED=1

# Per-stage timing (roi/clahe/infer/zones/assign/slot + the decode meter) is
# opt-in — flip it on at run time, no rebuild:
#   docker run -e PERF_TRACE=1 -e PERF_TRACE_EVERY=50 ...
# The effective-FPS summary is always on and needs nothing.
ENV PERF_TRACE=0

# Copy app code
COPY . .

EXPOSE 8000

# On start: seed the drawn zoning geometry (areas/slots/boundaries) into any
# EMPTY tables, then launch the API. `--if-empty` makes this a first-run seed —
# a restart/redeploy against a populated DB skips it and never clobbers live
# occupancy. Disable with SEED_GEOMETRY_ON_START=false; point at another dump
# with GEOMETRY_FILE=/path/to/dump.json. main.py still creates tables + seeds
# parking_areas from config.yaml, so the seed is additive (slots + boundaries).
ENTRYPOINT ["sh", "-c", "if [ \"${SEED_GEOMETRY_ON_START:-true}\" = true ] && [ -f \"${GEOMETRY_FILE:-geometry.json}\" ]; then echo '[entrypoint] seeding geometry into empty tables...'; python tools/sync_geometry.py seed --in \"${GEOMETRY_FILE:-geometry.json}\" --if-empty || echo '[entrypoint] geometry seed skipped (continuing)'; fi; exec python main.py --api"]