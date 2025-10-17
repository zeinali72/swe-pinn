#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting C++ Preprocessing Build and Run ---"

# --- 1. COMPILE THE C++ CODE ---
echo "[1/3] Navigating to scripts directory and creating build folder..."
# This ensures the script can be run from anywhere in the project
cd "$(dirname "$0")"
mkdir -p build
cd build

echo "[2/3] Configuring project with CMake..."
cmake .. > /dev/null # Hide verbose cmake output

echo "[3/3] Compiling the C++ code with make..."
make

# --- 2. RUN THE EXECUTABLE ---
echo -e "\n--- Running the C++ Preprocessor ---"
# Navigate back to the project root to ensure correct relative paths for data
cd ../.. 

# Define file paths
SCENARIO_NAME='one_building_DEM_zero'
BASE_PATH="data/${SCENARIO_NAME}"
ANGLE_FILE="${BASE_PATH}/2D zone_one_building_DEM_zero_DWF_angle2d.csv"
DEPTH_FILE="${BASE_PATH}/2D zone_one_building_DEM_zero_DWF_depth2d.csv"
SPEED_FILE="${BASE_PATH}/2D zone_one_building_DEM_zero_DWF_speed2d.csv"
OUTPUT_FILE="${BASE_PATH}/validation_tensor.bin" # C++ outputs a binary file

echo "Input Angle File: ${ANGLE_FILE}"
echo "Input Depth File: ${DEPTH_FILE}"
echo "Input Speed File: ${SPEED_FILE}"
echo "Output Binary File: ${OUTPUT_FILE}"

# Run the compiled C++ program from the root of the project
./scripts/build/preprocess "${ANGLE_FILE}" "${DEPTH_FILE}" "${SPEED_FILE}" "${OUTPUT_FILE}"

echo -e "\n--- C++ Preprocessing Finished Successfully ---"
