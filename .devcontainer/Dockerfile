# syntax=docker/dockerfile:1

# ---------- Base image ----------
FROM nvcr.io/nvidia/jax:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ---------- System packages (C++, cmake, Chrome for Kaleido) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    curl gnupg ca-certificates \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
       | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy helper scripts ----------
COPY install_dependencies.sh /workspace/install_dependencies.sh
COPY install_requirements.sh /workspace/install_requirements.sh
COPY requirements.txt        /workspace/requirements.txt

# ---------- GDAL header paths for pip builds ----------
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# ---------- Install OS + Python deps ----------
RUN chmod +x install_dependencies.sh install_requirements.sh && \
    ./install_dependencies.sh && \
    ./install_requirements.sh && \
    rm -rf /var/lib/apt/lists/*

# ---------- RTK-AI (token compression proxy for Claude Code) ----------
RUN curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/master/install.sh | sh \
    && ln -sf /root/.local/bin/rtk /usr/local/bin/rtk

# ---------- Gemini CLI ----------
RUN npm install -g @google/gemini-cli

CMD ["bash"]