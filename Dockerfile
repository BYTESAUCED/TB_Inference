# FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project definition
COPY pyproject.toml ./

# Install Python with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12

# Install dependencies with cache mount
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Command to run the application
CMD ["uv", "run","main.py"]
