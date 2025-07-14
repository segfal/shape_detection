cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(shape_recognizer)

# Set LibTorch path - adjust this to your actual LibTorch installation path
set(CMAKE_PREFIX_PATH "~/libs/libtorch")

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED PATHS /opt/homebrew/opt/opencv /usr/local/opt/opencv)

# Create executable
add_executable(shape_recognizer main.cpp)

# Link libraries
target_link_libraries(shape_recognizer "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Set C++ standard
set_property(TARGET shape_recognizer PROPERTY CXX_STANDARD 17)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Copy LibTorch libraries to build directory (for easier deployment)
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dylib")
add_custom_command(TARGET shape_recognizer
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:shape_recognizer>) 