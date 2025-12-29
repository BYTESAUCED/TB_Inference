# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - software-properties-common for add-apt-repository
# - glib/gl for opencv
# - curl for installing uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project definition
COPY pyproject.toml uv.lock ./

# Install dependencies (using system python 3.12)
# We need to explicitly tell uv to use python3.12 if default is different, strictly speaking UV_SYSTEM_PYTHON=1 uses found python.
RUN uv python install 3.12
RUN uv sync --frozen --no-dev --extra gpu

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Command to run the application
CMD ["uv", "run", "main.py"]
