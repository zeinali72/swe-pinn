#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential git openssh-client curl ca-certificates gnupg \
    gdal-bin libgdal-dev sudo tmux vim

# Install Node.js 20.x (required by Gemini CLI)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y --no-install-recommends nodejs

# Install GitHub CLI (available in Ubuntu universe repo)
apt-get install -y --no-install-recommends gh

apt-get clean
rm -rf /var/lib/apt/lists/*