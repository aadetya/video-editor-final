# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3-pip \
  ffmpeg \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libglib2.0-0 \
  libgl1-mesa-glx \
  git \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/audio_files \
  /app/clips \
  /app/fonts \
  /app/outputs \
  /app/static \
  /app/templates \
  /app/uploads

# Copy application files
COPY app.py .
COPY video_overlay_script.py .

# Copy directories
COPY fonts/ ./fonts/
COPY static/ ./static/
COPY templates/ ./templates/

# Copy default clips and audio (so RunPod has them too)
COPY data/clips/ ./clips/
COPY data/audio_files/ ./audio_files/

# Expose port for Flask app
EXPOSE 5000

# Set the entrypoint
CMD ["python3", "app.py"]
