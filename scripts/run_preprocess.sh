#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

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
# Assuming your CMakeLists.txt names the executable "preprocess_bin"
"$BUILD_DIR/preprocess_bin" "$@"