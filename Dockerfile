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

# Install Python build-time requirements
# We pin numpy < 2.0.0 because older ML libraries (like torchreid) often break on numpy 2.x
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy<2.0.0" Cython scipy torch torchvision

# Copy requirements and install to /install
COPY requirements.txt .

# Install torchreid separately to handle its build isolation/numpy requirements
# We use --no-build-isolation because we already installed numpy globally in this stage
RUN pip install --no-cache-dir --prefix=/install "numpy<2.0.0" && \
    pip install --no-cache-dir --prefix=/install --no-build-isolation "git+https://github.com/KaiyangZhou/deep-person-reid.git" && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


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

# Copy installed Python packages from builder to system path
COPY --from=builder /install /usr/local

# Copy app code (excluding files in .dockerignore)
COPY . .

EXPOSE 8000

# Using system python directly. 
# We run the migration script first, then start the main application in API mode.
ENTRYPOINT ["sh", "-c", "python migrate_parking_slots_to_db.py --import-json && python main.py --api"]
