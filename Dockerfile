# Start from an NVIDIA JAX image from 2023 (e.g., 23.08)
# This is compatible with the Tesla V100 (Volta architecture) GPU
FROM nvcr.io/nvidia/jax:23.08-py3

# Set the working directory and HOME env var to /workspace (OVHcloud standard)
ENV HOME=/workspace
WORKDIR /workspace

# Install git (bash is already in the base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first
COPY requirements.txt /workspace/requirements.txt

# Install your *additional* Python dependencies
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Copy your entire repository (your "workspace") into the image
COPY . /workspace

# Copy the job script and make it executable
COPY run_job.sh /workspace/run_job.sh
RUN chmod +x /workspace/run_job.sh

# CRITICAL: Grant ownership to the OVHcloud user (UID 42420)
RUN chown -R 42420:42420 /workspace

# Set the default command to run when the container starts
CMD ["/workspace/run_job.sh"]