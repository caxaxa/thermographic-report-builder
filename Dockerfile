# Multi-stage build for thermographic report builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels .

# ===== Final stage =====
FROM python:3.11-slim

# Install ONLY the minimal LaTeX packages we need
# This reduces image size from 5GB to ~800MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core LaTeX
    texlive-latex-base \
    # Portuguese support (babel-brazil)
    texlive-lang-portuguese \
    # Additional packages: geometry, fancyhdr, subfig, booktabs, xcolor, tikz
    texlive-latex-extra \
    # Ghostscript for PDF compression
    ghostscript \
    # OpenCV dependencies (minimal)
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user (non-root)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/report_work && \
    chown -R appuser:appuser /tmp/report_work

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml README.md /app/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create assets directory for logo images
RUN mkdir -p /app/assets && chown appuser:appuser /app/assets

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Entrypoint
ENTRYPOINT ["python", "-m", "thermographic_report_builder.main"]
