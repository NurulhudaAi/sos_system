# ==============================================================================
# Dockerfile for SOS Detection System & Model Server
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_ENV=production

# Configurable PyTorch wheel index (defaults to CPU; override with cu121 for NVIDIA GPU servers)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install system dependencies (OpenCV headless dependencies, FFmpeg, build tools for lap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and pre-install build prerequisites
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install PyTorch from configurable index URL (CPU by default, or CUDA via --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121)
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL}

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and default configuration
COPY . .

# Ensure runtime directories exist
RUN mkdir -p logs/snapshots alerts

# Expose Model Server (FastAPI) and Default Streaming port
EXPOSE 8000 8081

# Default command: launch the model server
CMD ["python3", "model_server.py"]
