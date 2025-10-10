# syntax=docker/dockerfile:1

# ---------- Base image ----------
FROM nvcr.io/nvidia/jax:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
RUN echo "Cache-Busted: $(date +%s)"

# ---------- Copy helper scripts ----------
COPY install_dependencies.sh   /workspace/install_dependencies.sh
COPY install_requirements.sh   /workspace/install_requirements.sh

# Optional: copy requirements; zero-byte file prevents COPY failure if absent
COPY requirements.txt          /workspace/requirements.txt

# ---------- Install OS + Python deps ----------
RUN chmod +x install_dependencies.sh install_requirements.sh && \
    ./install_dependencies.sh && \
    ./install_requirements.sh && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash"]
