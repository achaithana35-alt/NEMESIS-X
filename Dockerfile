FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

LABEL maintainer="Pari2003 <maitraiyeebhatt@gmail.com>"
LABEL description="Real-Time Video Anomaly Detection Pipeline"

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output directories
RUN mkdir -p outputs results weights data

# Default command
CMD ["python3", "-m", "src.pipeline", "--source", "0", "--config", "configs/default.yaml"]

# Expose Streamlit port
EXPOSE 8501

# Alternative: run the dashboard
# CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
