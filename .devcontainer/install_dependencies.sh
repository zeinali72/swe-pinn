#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential git openssh-client curl ca-certificates gnupg \
    gdal-bin libgdal-dev sudo tmux vim nodejs npm

apt-get clean
rm -rf /var/lib/apt/lists/*
