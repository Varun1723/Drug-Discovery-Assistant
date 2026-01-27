# ============================================================================
# Universal Dockerfile (NVIDIA GPU + CPU Fallback)
# Fixed: Added 'ipython_genutils' to fix stmol/ipywidgets error
# ============================================================================

# 1. Use Official PyTorch Image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. System Setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements
COPY requirements.txt .

# 5. Smart Dependency Installation
# STEP A: Clean requirements.txt
# We remove 'torch' (so we keep the GPU version) and 'deepchem'/'scipy' (to install them safely later)
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/scipy/d' requirements.txt && \
    sed -i '/deepchem/d' requirements.txt && \
    sed -i '/tensorflow/d' requirements.txt

# STEP B: Install General Dependencies WITH resolution
# This automatically installs pyarrow, toml, blinker, watchdog, etc.
RUN pip install --no-cache-dir -r requirements.txt

# STEP C: Install Scientific Stack & Legacy Fixes
# Added 'ipython_genutils' here to fix the stmol error
RUN pip install --no-cache-dir \
    "scipy<1.10" \
    "deepchem==2.7.1" \
    "tensorflow==2.13.0" \
    "ipython_genutils>=0.2.0"

# 6. Copy Application Code
COPY . .

# 7. Optimization Environment Variables
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6"
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"
# Fix Matplotlib cache warning by setting a writable directory
ENV MPLCONFIGDIR="/tmp/matplotlib"

# 8. Create User & Run
RUN groupadd -r drugdiscovery && useradd -r -g drugdiscovery drugdiscovery \
    && chown -R drugdiscovery:drugdiscovery /app
USER drugdiscovery

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]