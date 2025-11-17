# syntax=docker/dockerfile:1

# ---------- Base image ----------
FROM nvcr.io/nvidia/jax:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
RUN echo "Cache-Busted: $(date +%s)"

# ---------- START: MODIFICATIONS FOR C++ ----------
# Install C++ build essentials and cmake
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
# ---------- END: MODIFICATIONS FOR C++ ----------

# ---------- Copy helper scripts ----------
# CORRECTED: Paths are relative to the project root (the build context)
COPY install_dependencies.sh   /workspace/install_dependencies.sh
COPY install_requirements.sh   /workspace/install_requirements.sh
COPY requirements.txt          /workspace/requirements.txt

# ---------- Install OS + Python deps ----------
# This installs git, gdal, and all pip requirements
RUN chmod +x install_dependencies.sh install_requirements.sh && \
    ./install_dependencies.sh && \
    ./install_requirements.sh && \
    rm -rf /var/lib/apt/lists/*

# ---------- IMPORTANT ----------
# We are NOT copying the src/, optimisation/, etc. folders.
# This image will only contain the environment and dependencies.
# The code will be pulled from Git at runtime on OVH.

CMD ["bash"]