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
COPY install_dependencies.sh   /workspace/install_dependencies.sh
COPY install_requirements.sh   /workspace/install_requirements.sh

# Optional: copy requirements; zero-byte file prevents COPY failure if absent
COPY requirements.txt          /workspace/requirements.txt

# ---------- Install OS + Python deps ----------
# The original install_dependencies.sh can be simplified if build-essential is already installed above
RUN chmod +x install_dependencies.sh install_requirements.sh && \
    ./install_dependencies.sh && \
    ./install_requirements.sh && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash"]
