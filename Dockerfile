# Multi-stage build for thermographic report builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Clone thermal_parser to get the DJI SDK plugins
RUN git clone --depth 1 https://github.com/SanNianYiSi/thermal_parser.git /build/thermal_parser_repo

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels . && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels git+https://github.com/SanNianYiSi/thermal_parser.git

# ===== Final stage =====
FROM python:3.11-slim

# Install ONLY the minimal LaTeX packages we need
# This reduces image size from 5GB to ~800MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core LaTeX
    texlive-latex-base \
    # Portuguese and English support (babel-brazil, babel-english)
    texlive-lang-portuguese \
    texlive-lang-english \
    # Additional packages: geometry, fancyhdr, subfig, booktabs, xcolor, tikz
    texlive-latex-extra \
    # Fonts: lmodern, etc.
    texlive-fonts-recommended \
    lmodern \
    fonts-lmodern \
    # Ghostscript for PDF compression
    ghostscript \
    # OpenCV dependencies (minimal)
    libgl1 \
    libglib2.0-0 \
    # exiftool for thermal_parser (reads EXIF metadata from thermal images)
    libimage-exiftool-perl \
    # libgomp for DJI Thermal SDK (OpenMP parallel processing)
    libgomp1 \
    # GDAL for flight visualization (GeoTIFF parsing)
    gdal-bin \
    libgdal36 \
    # git is needed to resolve the solar-report-utils git+https dependency
    # when pip installs the package wheels in this runtime stage
    git \
    && mktexlsr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user (non-root)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/report_work && \
    chown -R appuser:appuser /tmp/report_work

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /build/wheels /tmp/wheels
# Remove numpy 2.4+ wheels (opencv-python requires numpy<2.3.0)
RUN rm -f /tmp/wheels/numpy-2.[4-9]*.whl 2>/dev/null || true
# --no-deps: /tmp/wheels already holds the full dependency tree (built by
# `pip wheel .`). Without it, pip rejects the local solar_report_utils wheel
# because the package metadata pins it as a git+https direct URL.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy DJI Thermal SDK plugins from thermal_parser repo
# These are required for temperature extraction from DJI R-JPEG images
COPY --from=builder /build/thermal_parser_repo/plugins /usr/local/lib/python3.11/site-packages/plugins

# Patch thermal_parser to use SDK v1.4 instead of v1.7 for compatibility with older M30T images
# SDK v1.7 returns error -16 for images captured before 2024
RUN sed -i 's/dji_thermal_sdk_v1.7_20241205/dji_thermal_sdk_v1.4_20220929/g' /usr/local/lib/python3.11/site-packages/thermal_parser/thermal.py

# Copy application code and set ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

# Install package in editable mode (deps already installed from wheels above)
RUN pip install --no-cache-dir --no-deps -e .

# Create assets directory for logo images
RUN mkdir -p /app/assets && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Entrypoint
ENTRYPOINT ["python", "-m", "thermographic_report_builder.main"]
