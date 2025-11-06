# =============================================
# LIGHTWEIGHT DOCKERFILE FOR AUTOENCODER PIPELINE
# Works well with 2 CPU cores and 8 GB RAM
# =============================================

FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal system deps (no build tools like gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install only lightweight Python deps first
COPY requirements.txt .

# Split heavy installs to reduce RAM peaks
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy pandas matplotlib seaborn tqdm scikit-learn && \
    pip install --no-cache-dir torch==2.8.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir scipy==1.15.1 notebook

# Copy project
COPY . .

# Set the default command
CMD ["python", "src/model_training.py"]

