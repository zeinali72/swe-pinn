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
# ---------- START: Install Google Chrome for Kaleido ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    && curl -fSsL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor | tee /usr/share/keyrings/google-chrome.gpg >> /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y \
    google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*
# ---------- END: Install Google Chrome for Kaleido ----------


# ---------- Copy helper scripts ----------
# CORRECTED: Paths are relative to the project root (the build context)
COPY install_dependencies.sh   /workspace/install_dependencies.sh
COPY install_requirements.sh   /workspace/install_requirements.sh
COPY requirements.txt          /workspace/requirements.txt

# >>> FIX: Define these ENV vars here so the build sees them <<<
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

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