cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(histogramCuda LANGUAGES CXX CUDA)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set source files
set(SOURCE_FILES src_cpp/Interface.cu)

# Specify the executable target
add_executable(histogramCuda ${SOURCE_FILES})

# Set the CUDA architecture (change it according to your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_61)

# Link OpenCV
target_link_libraries(histogramCuda PRIVATE ${OpenCV_LIBS})

# Specify the include directories for CUDA
target_include_directories(histogramCuda PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set the runtime library to use (change it according to your GPU)
set_property(TARGET histogramCuda PROPERTY CUDA_ARCHITECTURES 61)

# Set compiler flags for C++14
set_target_properties(histogramCuda PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

