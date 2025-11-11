# Minimal LaTeX compiler for PDF generation
FROM python:3.11-slim

# Install LaTeX packages + AWS CLI for S3 operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core LaTeX
    texlive-latex-base \
    # Portuguese support
    texlive-lang-portuguese \
    # Additional packages (geometry, fancyhdr, etc.)
    texlive-latex-extra \
    # Fonts (lmodern, etc.)
    texlive-fonts-recommended \
    lmodern \
    # Ghostscript for PDF compression
    ghostscript \
    # AWS CLI v2 dependencies
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Install boto3 for Python S3 operations
RUN pip install --no-cache-dir boto3

# Create working directory
WORKDIR /work

# Copy compilation script
COPY compile.py /app/compile.py

# Run as non-root
RUN useradd -m -u 1000 compiler && chown compiler:compiler /work
USER compiler

ENTRYPOINT ["python", "/app/compile.py"]
