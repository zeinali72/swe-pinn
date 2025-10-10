#!/usr/bin/env bash
set -euo pipefail

# Always upgrade pip first
python -m pip install --no-cache-dir --upgrade pip

REQ_FILE="/workspace/requirements.txt"
if [[ -s "${REQ_FILE}" ]]; then
    echo "Installing Python packages from ${REQ_FILE}"
    python -m pip install --no-cache-dir -r "${REQ_FILE}"
else
    echo "No requirements.txt (or it is empty) – skipping Python package install."
fi
