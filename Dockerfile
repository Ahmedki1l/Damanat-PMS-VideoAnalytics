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

# Create a virtual environment for the app
# This avoids --prefix issues and ensures a clean, portable environment.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python build-time requirements (CPU-optimized)
# We pin numpy < 2.0.0 for compatibility.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy<2.0.0" Cython scipy opencv-python-headless && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .

# Install torchreid separately to handle its build isolation requirements.
# Since we are in a venv, all dependencies installed above are available here.
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/KaiyangZhou/deep-person-reid.git" && \
    pip install --no-cache-dir -r requirements.txt


# =========================
# Stage 2: Runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (ODBC + OpenCV system libs)
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

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV SNAPSHOT_PATH=vehicle_images

# Copy app code (excluding files in .dockerignore)
COPY . .

EXPOSE 8000

# We run the migration script first, then start the main application in API mode.
ENTRYPOINT ["sh", "-c", "python migrate_parking_slots_to_db.py --import-json && python main.py --api"]
