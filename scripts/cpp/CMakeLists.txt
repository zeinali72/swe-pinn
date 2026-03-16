cmake_minimum_required(VERSION 3.10)

# Set the project name
project(PreprocessData)

# Set C++ standard to C++17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(preprocess_bin preprocess.cpp)

# --- NEW: Find and link the threads library ---
# This is required for std::thread and std::async
find_package(Threads REQUIRED)
target_link_libraries(preprocess_bin PRIVATE Threads::Threads)

# Allow the executable to be run from anywhere by installing it to a known location
install(TARGETS preprocess_bin DESTINATION bin)

