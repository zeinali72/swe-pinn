#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# =============================================================================
# InfoWorks ICM Data Preprocessing Pipeline
# =============================================================================
#
# Full pipeline:
#   1. CSV -> Binary:  cpp/preprocess.cpp  (this script builds & runs it)
#   2. Binary -> .npy: binary_to_numpy.py
#   3. .npy -> datasets: generate_training_data.py
#
# For gauge data from separate CSVs: process_gauge_csvs.py
#
# Column convention throughout: [t, x, y, h, u, v]
#   - t: time in seconds
#   - x, y: spatial coordinates
#   - h: water depth (m)
#   - u, v: velocity components (m/s), converted from speed + angle (radians)
#
# Usage:
#   ./run_preprocess.sh <angle_csv> <depth_csv> <speed_csv> <output_bin>
# =============================================================================

# 1. Define where the C++ source code now lives
CPP_SOURCE_DIR="$(dirname "$0")/cpp"
BUILD_DIR="${CPP_SOURCE_DIR}/build"

echo "--- Building C++ Preprocessor ---"

# 2. Create a build directory if it doesn't exist (keeps source clean)
mkdir -p "$BUILD_DIR"

# 3. Run CMake to configure the project
# -S points to source (scripts/cpp), -B points to build output (scripts/cpp/build)
cmake -S "$CPP_SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

# 4. Compile the code
cmake --build "$BUILD_DIR" --config Release

echo "--- Running Preprocessor ---"

# 5. Run the resulting executable
"$BUILD_DIR/preprocess_bin" "$@"
